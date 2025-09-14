import os
import csv
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from django.conf import settings
import numpy as np
import json


# --- Helper Function ---
def calculate_keyword_score(student_skills, internship_skills):
    student_set = set([skill.strip().lower() for skill in student_skills.split(',')])
    internship_set = set([skill.strip().lower() for skill in (internship_skills or "").split(',')])
    intersection = student_set.intersection(internship_set)
    union = student_set.union(internship_set)
    return 0.0 if not union else len(intersection) / len(union)


# --- Analytics Function ---
def get_analytics_for_internship(internship, user_skills_str):
    """
    Generates an explanation and learning roadmap for a specific internship.
    """
    # ** THE FIX IS HERE **
    # We move the Google imports inside the function. They will only be loaded
    # when this specific analytics function is called.
    import google.generativeai as genai
    from googleapiclient.discovery import build

    try:
        # 1. Configure APIs
        genai.configure(api_key=settings.GEMINI_API_KEY)
        youtube = build("youtube", "v3", developerKey=settings.YOUTUBE_API_KEY)

        # 2. Determine Lacking Skills
        user_skills_set = {s.strip().lower() for s in user_skills_str.split(',')}
        internship_skills_set = {s.strip().lower() for s in internship.get('Skills', '').split(',')}
        lacking_skills_set = internship_skills_set - user_skills_set

        internship_text = f"Title: {internship.get('Internship Title', '')}\nDescription: {internship.get('Internship Description', '')}"
        internship_skills_str = ", ".join(internship_skills_set)
        lacking_skills_str = ", ".join(lacking_skills_set)

        # 3. Build the Prompt for Gemini
        prompt = f"""
        You are an internship recommendation explainer.
        Given the following internship details and user information:
        - Internship Details: {internship_text}
        - All Skills Required: {internship_skills_str}
        - User's Skills: {user_skills_str}
        - User's Lacking Skills: {lacking_skills_str}

        Please perform the following tasks:
        1. Explain briefly why this internship is a good recommendation for the user.
        2. For each of the "User's Lacking Skills", create a detailed learning roadmap.

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

        # 4. Call the Generative Model
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        response_text = ""
        if hasattr(response, 'text') and response.text:
            response_text = response.text
        elif hasattr(response, 'parts') and response.parts:
            response_text = response.parts[0].text
        else:
            return {"error": "Invalid response format from Gemini API"}

        # Clean and parse the JSON response
        cleaned_response = response_text.strip().replace('```json', '').replace('```', '').strip()
        parsed_data = json.loads(cleaned_response)

        # 5. Augment with YouTube Videos
        for skill_roadmap in parsed_data.get("lacking_skills_roadmap", []):
            skill_name = skill_roadmap.get("skill_name")
            if not skill_name: continue

            request = youtube.search().list(q=f"{skill_name} tutorial playlist", part="snippet", type="playlist",
                                            maxResults=3)
            youtube_response = request.execute()

            videos = [{"title": item["snippet"]["title"],
                       "url": f"https://www.youtube.com/playlist?list={item['id']['playlistId']}"} for item in
                      youtube_response.get("items", [])]

            if "resources" not in skill_roadmap["learning_roadmap"]:
                skill_roadmap["learning_roadmap"]["resources"] = {}
            skill_roadmap["learning_roadmap"]["resources"]["youtube_playlists"] = videos

        return parsed_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to generate analytics: {str(e)}"}


# --- Recommendation Engine Class ---
class RecommendationEngine:
    def __init__(self):
        print("Loading recommendation engine assets...")
        asset_path = os.path.join(settings.BASE_DIR, 'recommender', 'ml_assets')
        model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(os.path.join(asset_path, 'internships.faiss'))

        with open(os.path.join(asset_path, 'index_to_id.pkl'), 'rb') as f:
            self.index_to_id_map = pickle.load(f)

        self.all_internships_map = {}
        column_names = ['id', 'Title', 'Locations', 'Skills', 'Description']
        with open(os.path.join(asset_path, 'internships.csv'), mode='r', encoding='utf-8') as csvfile:
            next(csvfile)
            reader = csv.DictReader(csvfile, fieldnames=column_names)
            for row in reader:
                self.all_internships_map[row['id']] = row
        print("✅ Recommendation engine loaded successfully.")

    def find_recommendations(self, skills, location, interest):
        if interest:
            student_text = f"A student with key skills in: {skills}. They are interested in an internship where they can {interest}."
        else:
            student_text = f"A student with key skills in: {skills}."

        student_embedding = self.model.encode([student_text], convert_to_numpy=True)
        faiss.normalize_L2(student_embedding)

        k = 200
        distances, indices = self.index.search(student_embedding, k)

        all_top_candidates = []
        user_explicit_skills = [s.strip().lower() for s in skills.split(',')]

        for i, idx in enumerate(indices[0]):
            if idx == -1: continue

            internship_id = self.index_to_id_map[idx]
            internship = self.all_internships_map[internship_id]
            sem_score = distances[0][i]

            skill_boost_score = 0
            internship_skills_text = internship.get('Skills', '').lower()
            for skill in user_explicit_skills:
                if skill in internship_skills_text:
                    skill_boost_score += 1

            normalized_boost = skill_boost_score / len(user_explicit_skills) if user_explicit_skills else 0
            final_score = (0.7 * sem_score) + (0.3 * normalized_boost)
            all_top_candidates.append({'final_score': final_score, 'internship': internship})

        all_top_candidates.sort(key=lambda x: x['final_score'], reverse=True)

        recommendations_in_location = [
            rec for rec in all_top_candidates
            if rec['internship']['Locations'].lower().strip() == location.lower()
        ]

        return recommendations_in_location, all_top_candidates