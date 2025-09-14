import os
import csv
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from django.conf import settings
import numpy as np
import google.generativeai as genai
from googleapiclient.discovery import build
import json

# This helper function calculates a simple keyword overlap score
def calculate_keyword_score(student_skills, internship_skills):
    student_set = set([skill.strip().lower() for skill in student_skills.split(',')])
    internship_set = set([skill.strip().lower() for skill in (internship_skills or "").split(',')])
    intersection = student_set.intersection(internship_set)
    union = student_set.union(internship_set)
    return 0.0 if not union else len(intersection) / len(union)


# ... (existing code for RecommendationEngine, etc.) ...

# --- NEW ANALYTICS FUNCTION ---
def get_analytics_for_internship(internship, user_skills_str):
    """
    Generates an explanation and learning roadmap for a specific internship.
    """
    try:
        # 1. Configure APIs using keys from settings.py
        genai.configure(api_key=settings.GEMINI_API_KEY)
        youtube = build("youtube", "v3", developerKey=settings.YOUTUBE_API_KEY)

        # 2. Determine Lacking Skills (make it dynamic)
        user_skills_set = {s.strip().lower() for s in user_skills_str.split(',')}
        internship_skills_set = {s.strip().lower() for s in internship.get('Skills', '').split(',')}
        lacking_skills_set = internship_skills_set - user_skills_set

        # Format for the prompt
        internship_text = f"Title: {internship.get('Internship Title', '')}\nDescription: {internship.get('Internship Description', '')}"
        internship_skills_str = ", ".join(internship_skills_set)
        lacking_skills_str = ", ".join(lacking_skills_set)

        # 3. Build the Dynamic Prompt for Gemini
        prompt = f"""
        You are an internship recommendation explainer.
        Given the following internship details and user information:
        - Internship Details: {internship_text}
        - All Skills Required: {internship_skills_str}
        - User's Skills: {user_skills_str}
        - User's Lacking Skills: {lacking_skills_str}

        Please perform the following tasks:
        1. Explain briefly why this internship is a good recommendation for the user.
        2. For each of the "User's Lacking Skills", create a detailed learning roadmap. Each roadmap must include:
           - 3 distinct learning milestones.
           - 2 relevant mini-project ideas.
           - 3 resources: 2 free online course links (preferably from NPTEL, otherwise others) and 1 link to the official documentation.

        Return the answer ONLY in the following JSON format. Do not add any text or explanations outside of the JSON structure.

        {{
          "explanation": "string",
          "lacking_skills_roadmap": [
            {{
              "skill_name": "string",
              "learning_roadmap": {{
                "milestones": ["string", "string", "string"],
                "mini_projects": ["string", "string"],
                "resources": {{
                  "online_courses": [
                    {{"name": "Course Name 1", "url": "https://course_url_1"}},
                    {{"name": "Course Name 2", "url": "https://course_url_2"}}
                  ],
                  "official_docs": "https://docs_url"
                }}
              }}
            }}
          ]
        }}
        """

        # 4. Call the Generative Model with better error handling
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        # Better response handling
        response_text = ""
        if hasattr(response, 'text') and response.text:
            response_text = response.text
        elif hasattr(response, 'parts') and response.parts:
            response_text = response.parts[0].text
        else:
            print("❌ Unexpected Gemini API response format")
            print(f"Full response: {response}")
            return {"error": "Invalid response format from Gemini API"}

        print("--- RAW GEMINI RESPONSE ---")
        print(f"Response text: {response_text}")
        print("---------------------------")

        # Clean the response text (remove any markdown formatting)
        cleaned_response = response_text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # Remove ```
        cleaned_response = cleaned_response.strip()

        # Try to parse JSON with better error handling
        try:
            parsed_data = json.loads(cleaned_response)
        except json.JSONDecodeError as json_error:
            print(f"❌ JSON parsing failed: {json_error}")
            print(f"❌ Attempted to parse: {cleaned_response[:500]}...")  # Show first 500 chars
            return {"error": f"Failed to parse Gemini response as JSON: {str(json_error)}"}

        # 5. Validate the parsed data structure
        if not isinstance(parsed_data, dict):
            return {"error": "Invalid response structure from Gemini API"}

        if "lacking_skills_roadmap" not in parsed_data:
            parsed_data["lacking_skills_roadmap"] = []

        # 6. Augment with YouTube Videos (with error handling)
        try:
            for skill_roadmap in parsed_data.get("lacking_skills_roadmap", []):
                skill_name = skill_roadmap.get("skill_name", "")
                if not skill_name:
                    continue

                request = youtube.search().list(
                    q=f"{skill_name} tutorial playlist",
                    part="snippet",
                    type="playlist",
                    maxResults=3
                )
                youtube_response = request.execute()

                videos = []
                for item in youtube_response.get("items", []):
                    title = item["snippet"]["title"]
                    playlist_id = item["id"]["playlistId"]
                    url = f"https://www.youtube.com/playlist?list={playlist_id}"
                    videos.append({"title": title, "url": url})

                # Initialize resources if it doesn't exist
                if "learning_roadmap" not in skill_roadmap:
                    skill_roadmap["learning_roadmap"] = {}
                if "resources" not in skill_roadmap["learning_roadmap"]:
                    skill_roadmap["learning_roadmap"]["resources"] = {}

                # Add videos to the roadmap
                skill_roadmap["learning_roadmap"]["resources"]["youtube_playlists"] = videos

        except Exception as youtube_error:
            print(f"⚠️ YouTube API error (non-fatal): {youtube_error}")
            # Continue without YouTube data

        return parsed_data

    except Exception as e:
        print(f"❌ Error in analytics generation: {e}")
        import traceback
        traceback.print_exc()  # This will show the full stack trace
        return {"error": f"Failed to generate analytics: {str(e)}"}

class RecommendationEngine:
    def __init__(self):
        print("Loading recommendation engine assets...")
        asset_path = os.path.join(settings.BASE_DIR, 'recommender', 'ml_assets')
        model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

        # Load all assets on startup
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(os.path.join(asset_path, 'internships.faiss'))

        with open(os.path.join(asset_path, 'index_to_id.pkl'), 'rb') as f:
            self.index_to_id_map = pickle.load(f)

        # Load all internship data into memory
        self.all_internships_map = {}
        column_names = ['id', 'Title', 'Locations', 'Skills', 'Description']
        with open(os.path.join(asset_path, 'internships.csv'), mode='r', encoding='utf-8') as csvfile:
            next(csvfile)
            reader = csv.DictReader(csvfile, fieldnames=column_names)
            for row in reader:
                self.all_internships_map[row['id']] = row
        print("✅ Recommendation engine loaded successfully.")

    def find_recommendations(self, skills, location, interest):
        # 1. Create a rich semantic query from all user inputs
        if interest:
            student_text = f"A student with key skills in: {skills}. They are interested in an internship where they can {interest}."
        else:
            student_text = f"A student with key skills in: {skills}."

        student_embedding = self.model.encode([student_text], convert_to_numpy=True)
        faiss.normalize_L2(student_embedding)

        # 2. Perform the initial broad semantic search with FAISS
        k = 200
        distances, indices = self.index.search(student_embedding, k)

        all_top_candidates = []
        # Create a clean list of the user's explicit skills for boosting
        user_explicit_skills = [s.strip().lower() for s in skills.split(',')]

        for i, idx in enumerate(indices[0]):
            if idx == -1: continue

            internship_id = self.index_to_id_map[idx]
            internship = self.all_internships_map[internship_id]

            sem_score = distances[0][i]

            # 3. **Aggressive Skill Boost Calculation**
            # This is the key logic: count how many of the user's skills are explicitly in the internship listing
            skill_boost_score = 0
            internship_skills_text = internship.get('Skills', '').lower()
            for skill in user_explicit_skills:
                if skill in internship_skills_text:
                    skill_boost_score += 1  # Add a point for each matching skill

            # Normalize the boost score by the number of skills the user provided
            normalized_boost = skill_boost_score / len(user_explicit_skills) if user_explicit_skills else 0

            # 4. **Final Scoring**
            # The semantic score provides the general relevance, but the skill boost has a strong influence
            final_score = (0.7 * sem_score) + (0.3 * normalized_boost)

            all_top_candidates.append({'final_score': final_score, 'internship': internship})

        # 5. Sort by the new, more balanced final score
        all_top_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        # Filter the final ranked list by location
        recommendations_in_location = [
            rec for rec in all_top_candidates
            if rec['internship']['Locations'].lower().strip() == location.lower()
        ]

        return recommendations_in_location, all_top_candidates


# Instantiate the engine ONCE when Django starts
engine = RecommendationEngine()