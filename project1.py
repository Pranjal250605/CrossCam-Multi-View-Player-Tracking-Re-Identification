import cv2, numpy as np, easyocr
from ultralytics import YOLO
from clip_extractor import CLIPFeatureExtractor
from sklearn.cluster import KMeans
from collections import deque
from statistics import median
from mediapipe.python.solutions import pose as mp_pose
from scipy.optimize import linear_sum_assignment


USE_REID   = True
USE_OCR    = True
POSE_EVERY = 1

BROADCAST_PATH = "broadcast.mp4"
TACTICAM_PATH  = "tacticam.mp4"
OUTPUT_PATH    = "cross_view_output1.mp4"
YOLO_MODEL     = "best.pt"
DEVICE         = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() else "cpu"

ROLL_WINDOW = 50
SYNC_WINDOW = 2
CLIP_THRESH = 0.8
SIM_THRESH  = 1.2
MAX_MEM_AGE = 100
TRACKLET    = 30

W_CLIP,W_POSE,W_HUE,W_POS,W_DEN = 1.0,0.5,0.4,0.3,0.2

mpPose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def pose_vec(img: np.ndarray):
    if img.size == 0:
        return np.zeros(99, dtype=np.float32)
    res = mpPose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return np.zeros(99, dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.visibility] for lm in res.pose_landmarks.landmark], dtype=np.float32).flatten()



def extract_jnum(crop, reader):
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray, detail=0)
        candidates = [r for r in results if r.isdigit() and 0 < len(r) <= 2]
        return int(candidates[0]) if candidates else None
    except:
        return None

class CrossCameraMapper:
    def __init__(self):
        self.cap_b = cv2.VideoCapture(BROADCAST_PATH)
        self.cap_t = cv2.VideoCapture(TACTICAM_PATH)

        self.det  = YOLO(YOLO_MODEL)
        self.clip = CLIPFeatureExtractor(device=DEVICE)
        self.ocr  = easyocr.Reader(['en'], gpu=(DEVICE=='cuda'))

        self.team_centroids = None
        self.off_hist = deque(maxlen=ROLL_WINDOW)
        self.cur_off = 0
        self.mem = {}
        self.next_tid = 100
        self.jersey_tid = {}
        self.id_registry = {}

    def detect(self, frame):
        res = self.det(frame, verbose=False)[0]
        return [tuple(map(int, b.tolist())) for b, c in zip(res.boxes.xyxy.cpu(), res.boxes.cls.cpu()) if int(c) == 2]

    def team_map(self, feats):
        if len(feats) < 2: return {id(f): 0 for f in feats}
        hues = np.array([[f['hue']] for f in feats], np.float32)
        k = KMeans(2, n_init=5, random_state=0).fit(hues)
        lbl, ctr = k.labels_, k.cluster_centers_.flatten()
        if self.team_centroids is None:
            order = np.argsort(ctr); self.team_centroids = ctr[order]
            map_ = {old: new for new, old in enumerate(order)}
        else:
            map_ = {i: int(np.argmin(np.abs(self.team_centroids - c))) for i, c in enumerate(ctr)}
        return {id(f): map_[lbl[i]] for i, f in enumerate(feats)}

    @staticmethod
    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def score(self, a, b):
        s = W_CLIP * self._cos(a['emb'], b['emb'])
        s += W_POSE * (1 - np.linalg.norm(a['pose'] - b['pose']) / 20)
        s += W_HUE * (1 - abs(a['hue'] - b['hue']))
        s += W_POS * (1 - np.linalg.norm(a['pos'] - b['pos']))
        s += W_DEN * (1 - abs(a['den'] - b['den']) / 5)
        if a['jnum'] is not None and b['jnum'] is not None and a['jnum'] == b['jnum']:
            s += 3
        return s

    def enforce_unique_ids(self, ids, feats):
        seen = set()
        for i, tid in enumerate(ids):
            if tid < 0:
                continue
            if tid in seen:
                ids[i] = -1
            else:
                seen.add(tid)
        return seen



    def _update(self, tid, f, tm):
        key = (f['view'], *f['bbox'])
        self.id_registry[key] = tid
        if f['jnum'] is not None:
            self.jersey_tid.setdefault(f['jnum'], tid)
        m = self.mem.setdefault(tid, dict(buf=deque(maxlen=TRACKLET), team=tm, last=-1))
        m['buf'].append(f)
        m['last'] = f['frame']

    def _match_registry(self, f):
        for key, tid in self.id_registry.items():
            v, x1, y1, x2, y2 = key
            if f['view'] != v: continue
            fx1, fy1, fx2, fy2 = f['bbox']
            if abs(x1 - fx1) < 10 and abs(y1 - fy1) < 10 and abs(x2 - fx2) < 10 and abs(y2 - fy2) < 10:
                return tid
        return -1
    
    def _match_memory(self, f, team):
        best_sim = -1
        best_tid = -1
        for tid, mem in self.mem.items():
            if mem['team'] != team or f['frame'] - mem['last'] > MAX_MEM_AGE:
                continue
            emb_avg = np.mean([m['emb'] for m in mem['buf']], axis=0)
            sim = self._cos(emb_avg, f['emb'])
            if sim > best_sim:
                best_sim = sim
                best_tid = tid
        return best_tid if best_sim > CLIP_THRESH else -1


    def run(self):
        fps = min(self.cap_b.get(5), self.cap_t.get(5))
        buf = [self.cap_t.read()[1] for _ in range(SYNC_WINDOW * 2 + 1)]
        writer = None; f = 0

        while True:
            ok, fb = self.cap_b.read()
            if not ok: break
            buf.pop(0); buf.append(self.cap_t.read()[1])

            pb = self.detect(fb)
            crops_b = [fb[y1:y2, x1:x2] for x1, y1, x2, y2 in pb]
            emb_b = self.clip.extract_batch(crops_b)
            feats_b = []
            for (x1, y1, x2, y2), e in zip(pb, emb_b):
                crop = fb[y1:y2, x1:x2]
                jnum = extract_jnum(crop, self.ocr) if USE_OCR else None
                feats_b.append(dict(
                    bbox=(x1, y1, x2, y2), emb=e, jnum=jnum,
                    pose=pose_vec(crop),
                    hue=np.mean(crop[:, :, 0]) / 255,
                    pos=np.array([(x1 + x2) / (2 * fb.shape[1]), (y1 + y2) / (2 * fb.shape[0])]),
                    den=(x2 - x1) * (y2 - y1) / (fb.shape[0] * fb.shape[1]),
                    view='b', frame=f))

            best_off = -SYNC_WINDOW - 1; best = -1
            for o, ft in enumerate(buf):
                if ft is None: continue
                pt = self.detect(ft)
                crops_t = [ft[y1:y2, x1:x2] for x1, y1, x2, y2 in pt]
                if not crops_t: continue
                emb_t = self.clip.extract_batch(crops_t)
                sim = sum(max(self._cos(e_b[:768], e_t[:768]) for e_t in emb_t) for e_b in emb_b)
                if sim > best: best, best_off = sim, o - SYNC_WINDOW
            self.off_hist.append(best_off)
            self.cur_off = int(median(self.off_hist))
            ft = buf[SYNC_WINDOW + self.cur_off]; pt = self.detect(ft)
            crops_t = [ft[y1:y2, x1:x2] for x1, y1, x2, y2 in pt]
            emb_t = self.clip.extract_batch(crops_t)
            feats_t = []
            for (x1, y1, x2, y2), e in zip(pt, emb_t):
                crop = ft[y1:y2, x1:x2]
                jnum = extract_jnum(crop, self.ocr) if USE_OCR else None
                feats_t.append(dict(
                    bbox=(x1, y1, x2, y2), emb=e, jnum=jnum,
                    pose=pose_vec(crop),
                    hue=np.mean(crop[:, :, 0]) / 255,
                    pos=np.array([(x1 + x2) / (2 * ft.shape[1]), (y1 + y2) / (2 * ft.shape[0])]),
                    den=(x2 - x1) * (y2 - y1) / (ft.shape[0] * ft.shape[1]),
                    view='t', frame=f + self.cur_off))

            tm = self.team_map(feats_b + feats_t)
            ids_b = [-1] * len(feats_b)
            ids_t = [-1] * len(feats_t)
            if len(feats_b) == 0 or len(feats_t) == 0:
                f += 1
                continue

            scores = np.zeros((len(feats_b), len(feats_t)))
            for i, a in enumerate(feats_b):
                for j, b_ in enumerate(feats_t):
                    if tm[id(a)] != tm[id(b_)]:
                        scores[i][j] = -9999
                    else:
                        scores[i][j] = self.score(a, b_)

            if len(feats_b) > 0 and len(feats_t) > 0:
                cost = -scores  # for maximization
                row_ind, col_ind = linear_sum_assignment(cost)

                for i, j in zip(row_ind, col_ind):
                    if scores[i][j] < SIM_THRESH:
                        continue

                    fb_i = feats_b[i]
                    ft_j = feats_t[j]

                    tid = -1
                    if fb_i['jnum'] is not None and fb_i['jnum'] in self.jersey_tid:
                        tid = self.jersey_tid[fb_i['jnum']]
                    elif ft_j['jnum'] is not None and ft_j['jnum'] in self.jersey_tid:
                        tid = self.jersey_tid[ft_j['jnum']]
                    if tid == -1:
                        tid = self._match_memory(fb_i, tm[id(fb_i)])
                    if tid == -1:
                        tid = self._match_memory(ft_j, tm[id(ft_j)])
                    if tid == -1:
                        tid = self.next_tid
                        self.next_tid += 1


                    ids_b[i] = ids_t[j] = tid
                    self._update(tid, fb_i, tm[id(fb_i)])
                    self._update(tid, ft_j, tm[id(ft_j)])

            used_ids_b = self.enforce_unique_ids(ids_b, feats_b)
            used_ids_t = self.enforce_unique_ids(ids_t, feats_t)


            for feats, ids, used_ids in ((feats_b, ids_b, used_ids_b), (feats_t, ids_t, used_ids_t)):
                for k, fv in enumerate(feats):
                    if ids[k] == -1:
                        if fv['jnum'] is not None and fv['jnum'] in self.jersey_tid:
                            tid = self.jersey_tid[fv['jnum']]
                        else:
                            # Priority 2: temporal memory
                            tid = self._match_memory(fv, tm[id(fv)])
                        if tid == -1 or tid in used_ids:
                            tid = self.next_tid
                            self.next_tid += 1
                        used_ids.add(tid)
                        ids[k] = tid
                        self._update(tid, fv, tm[id(fv)])



            canvas = np.hstack([fb, ft])
            for feats, ids, off in ((feats_b, ids_b, 0), (feats_t, ids_t, fb.shape[1])):
                for fv, tid in zip(feats, ids):
                    x1, y1, x2, y2 = fv['bbox']
                    col = (0, 255, 0) if self.mem[tid]['team'] == 0 else (255, 0, 0)
                    label = f"{tid}"
                    if fv['jnum'] is not None:
                        label += f"#{fv['jnum']}"
                    cv2.rectangle(canvas, (x1 + off, y1), (x2 + off, y2), col, 2)
                    cv2.putText(canvas, label, (x1 + off, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            if writer is None:
                writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            if f % 10 == 0:
                print(f"[f={f}] raw {best_off:+d} med {self.cur_off:+d}")
            f += 1

        self.cap_b.release(); self.cap_t.release(); writer.release()
        print("DONE →", OUTPUT_PATH)

if __name__ == "__main__":
    CrossCameraMapper().run()
