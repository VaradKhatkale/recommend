from rest_framework.views import APIView
from rest_framework.response import Response
from django.views.generic import TemplateView
from django.core.cache import cache  # <-- Import Django's cache
from .engine import engine
from .serializers import RecommendationSerializer
from .engine import get_analytics_for_internship

class HomePageView(TemplateView):
    template_name = "home.html"

class PortalPageView(TemplateView):
    template_name = "portal.html"


class InternshipAnalyticsView(APIView):
    def get(self, request, *args, **kwargs):
        internship_id = request.query_params.get('id', '')
        user_skills = request.query_params.get('skills', '')

        print(f"📊 Analytics request - ID: {internship_id}, Skills: {user_skills}")

        if not internship_id or not user_skills:
            return Response({"error": "Internship ID and user skills are required."}, status=400)

        # Fetch the specific internship details from the engine's map
        internship = engine.all_internships_map.get(internship_id)
        if not internship:
            print(f"❌ Internship not found for ID: {internship_id}")
            return Response({"error": "Internship not found."}, status=404)

        print(f"✅ Found internship: {internship.get('Title', 'Unknown')}")

        # Call the analytics function with better error handling
        try:
            analytics_data = get_analytics_for_internship(internship, user_skills)

            # Check if the response contains an error
            if isinstance(analytics_data, dict) and "error" in analytics_data:
                print(f"❌ Analytics function returned error: {analytics_data['error']}")
                return Response(analytics_data, status=500)

            print("✅ Analytics generated successfully")
            return Response(analytics_data)

        except Exception as e:
            print(f"❌ Unexpected error in analytics view: {e}")
            import traceback
            traceback.print_exc()
            return Response({"error": f"Internal server error: {str(e)}"}, status=500)
class RecommendInternships(APIView):
    def get(self, request, *args, **kwargs):
        skills = request.query_params.get('skills', '')
        location = request.query_params.get('location', '')
        interest = request.query_params.get('interest', '')

        if not skills or not location:
            return Response({"error": "Please provide both 'skills' and 'location' parameters."}, status=400)

        # --- CACHING LOGIC ---
        # 1. Create a unique key for this request
        cache_key = f"rec_{skills}_{location}_{interest}"

        # 2. Try to get the result from the cache first
        cached_result = cache.get(cache_key)
        if cached_result:
            print("✅ Returning response from cache!")
            return Response(cached_result)

        # 3. If not in cache, calculate the result as before
        print("❌ Cache miss. Calculating new recommendations...")
        in_location_recs, global_recs = engine.find_recommendations(skills, location, interest)

        # ... (rest of your fallback and filtering logic is the same)
        MINIMUM_SCORE_THRESHOLD = 0.55
        FALLBACK_COUNT = 3

        good_in_location = [rec for rec in in_location_recs if rec['final_score'] >= MINIMUM_SCORE_THRESHOLD]
        good_global = [rec for rec in global_recs if rec['final_score'] >= MINIMUM_SCORE_THRESHOLD]

        results = []
        message = ""

        if good_in_location:
            message = f"Top matches in your preferred location: {location}"
            results = good_in_location[:5]
        elif good_global:
            message = f"No ideal matches found in {location}. Here are the best matches from other locations:"
            results = good_global[:FALLBACK_COUNT]
        else:
            message = f"No ideal matches were found anywhere. Here are the top {FALLBACK_COUNT} closest skill matches from all locations:"
            results = global_recs[:FALLBACK_COUNT]

        serializer = RecommendationSerializer(results, many=True)
        final_response_data = {
            "message": message,
            "recommendations": serializer.data
        }

        # 4. Save the new result to the cache for 1 hour (3600 seconds)
        cache.set(cache_key, final_response_data, timeout=3600)

        return Response(final_response_data)