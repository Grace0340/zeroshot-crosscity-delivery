"""Check that IMPEL's Llama-3 embeddings align region-by-region with our
processed demand tensors."""
import numpy as np
import yaml

cfg = yaml.safe_load(open("configs/default.yaml"))

for c in ["SH", "HZ", "CQ", "YT", "JL"]:
    emb = np.load(f"{cfg['model']['llm_emb_path']}/llmvec_llama3_Delivery_{c}.npy")
    ours = np.load(f"{cfg['data']['processed_dir']}/{c.lower()}.npz", allow_pickle=True)
    n_ours = ours["demand"].shape[1]
    status = "OK" if emb.shape[0] == n_ours else "MISMATCH"
    print(f"{c}: impel_emb={emb.shape} ours_regions={n_ours} regions_head={ours['regions'][:6]} -> {status}")
