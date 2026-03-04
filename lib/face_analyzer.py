import insightface

face_analyzer = insightface.app.FaceAnalysis(
    name="buffalo_m",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    allowed_modules=["detection", "recognition"],
)
face_analyzer.prepare(ctx_id=0, det_size=(448, 448))
