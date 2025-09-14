web: gunicorn recommendation_project.wsgi --workers 4 --threads 4 --worker-class gthread --log-level debug
release: python manage.py migrate