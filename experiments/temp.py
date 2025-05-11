from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# 1. event 파일 로드 (경로를 네 로그 폴더에 맞게 수정)
ea = event_accumulator.EventAccumulator("runs/modelA")
ea.Reload()

# 2. 특정 태그 추출 (예: "loss/train")
events = ea.Scalars("loss/train")

# 3. 데이터 프레임으로 변환
df = pd.DataFrame(events)
df = df.rename(columns={"wall_time": "time", "step": "epoch", "value": "loss"})

# 4. 저장하거나 시각화
df.to_csv("train_loss_modelA.csv", index=False)
