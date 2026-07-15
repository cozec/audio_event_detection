Start with YAMNet + ESC-50, then turn it into an edge-oriented project:

1. Select 5–10 ESC-50 classes.
2. Use YAMNet embeddings and fine-tune its classifier.
3. Build streaming inference with overlapping audio windows.
4. Export to TFLite.
5. Apply INT8 quantization.
6. Compare accuracy, latency, RAM, model size, and false-positive rate.
7. Add a macbook inference demo.
8. Then compare it with a small custom DS-CNN.