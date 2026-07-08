FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate_category_trends.py compute_category_trends.py generate_data.py train_shadow_buyer.py train_ripple.py app.py ./
COPY static ./static

# Build the full pipeline at image build time, in order, so the container
# starts up instantly with trained models already saved to disk.
RUN python generate_category_trends.py \
 && python compute_category_trends.py \
 && python generate_data.py \
 && python train_shadow_buyer.py \
 && python train_ripple.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
