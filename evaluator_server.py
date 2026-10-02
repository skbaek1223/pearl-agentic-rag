import argparse
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, default="0", help="Single GPU id for the evaluator.")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Defaults to modernbert_evaluator.DEFAULT_CHECKPOINT.")
    p.add_argument("--threshold", type=float, default=None,
                   help="P(sufficient) cutoff. Defaults to modernbert_evaluator.SUFFICIENT_THRESHOLD (0.60).")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from modernbert_evaluator import ModernBertEvaluator, DEFAULT_CHECKPOINT, SUFFICIENT_THRESHOLD

    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    threshold = args.threshold if args.threshold is not None else SUFFICIENT_THRESHOLD
    print(f"[evaluator_server] loading {checkpoint} on cuda:0 (physical GPU {args.gpu}), threshold={threshold}...")
    evaluator = ModernBertEvaluator(checkpoint, device="cuda:0", threshold=threshold)
    print("[evaluator_server] ready")

    class State(BaseModel):
        question: str
        recent_reasoning: str
        search_query: str
        extracted_info: str

    class PredictBatchRequest(BaseModel):
        states: list[State]

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "checkpoint": checkpoint, "threshold": threshold}

    @app.post("/predict_batch")
    def predict_batch(req: PredictBatchRequest):
        states = [s.model_dump() for s in req.states]
        results = evaluator.predict_batch(states)
        return {"results": [[bool(is_suff), float(conf)] for is_suff, conf in results]}

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
