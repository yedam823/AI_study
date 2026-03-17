import gymnasium as gym
import torch
import torch.nn as nn
import time

# 1. 코랩에서 썼던 것과 똑같은 신경망 뼈대 준비
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)

# 2. 모델 뼈대 생성 및 다운로드한 '뇌' 이식하기
model = DQN()
model.load_state_dict(torch.load('cartpole_dqn.pth'))
model.eval() # 학습 모드가 아닌 평가(실행) 모드로 전환

# 3. 사람이 볼 수 있는 창 띄우기 설정 ('human')
env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()
score = 0

print("🚀 AI 제어 시작!")

# 4. 실시간 제어 루프
while True:
    # 현재 상태를 텐서로 변환하여 모델에 입력
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    # 모델이 가장 좋다고 생각하는 행동(0 또는 1) 선택 (Epsilon=0 상태와 동일)
    action = model(state_tensor).argmax().item()
    
    # 행동 실행
    state, reward, terminated, truncated, _ = env.step(action)
    score += 1
    
    # 너무 빨리 지나가면 보기 힘드므로 살짝 딜레이 (선택 사항)
    time.sleep(0.02)
    
    if terminated or truncated:
        print(f"🏁 테스트 종료! 최종 점수: {score}")
        time.sleep(1) # 종료 전 1초 대기
        break

env.close()