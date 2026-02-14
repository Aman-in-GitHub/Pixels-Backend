import insightface

face_analyzer = insightface.app.FaceAnalysis(
    name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
