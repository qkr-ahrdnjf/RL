# 🎬 강화학습 기반 영화 추천 시스템

##  주요 기능

| ε-Greedy, Thompson Sampling 구현 |
| Sequential DQN | LSTM 기반 시퀀셜 추천 모델 |
| 하이퍼파라미터 튜닝 | Grid Search 기반 최적화 |
| 다중 시드 실험 | 5개 시드로 통계적 신뢰성 확보 |
| TMDB API 연동 | 포스터, 줄거리, 평점 정보 제공 |
| 모델 저장/로드 | 학습된 DQN 모델 재사용 가능 |

---

## 🛠 설치 방법

https://github.com/qkr-ahrdnjf/RL
학습된 모델은 'models/dqn_model.pth'에 포함되어 있음.

### 3. TMDB API 키 발급
영화 포스터와 줄거리를 보려면 TMDB API 키가 필요하다.
1. https://www.themoviedb.org/ 에서 회원가입
2. Settings → API → Create API Key
3. 코드의 `TMDB_API_KEY` 변수에 입력

---

## 🚀 실행 방법

실행 환경은 requirements.txt에 기재함

### Google Colab에서 실행
1. `movie_recommender_complete.ipynb` 파일을 Colab에 업로드
2. 런타임 → 런타임 유형 변경 → GPU 선택
3. 셀을 순서대로 실행

### 학습된 모델 사용하기
이미 학습된 모델을 사용하려면:

Part 3 코드 두 줄을 다음으로 바꾼다.
final_dqn = SequentialDQNRecommender(env)
final_dqn.train(n_episodes = 500, steps_per_episode = 10)
->
final_dqn = SequentialDQNRecommender(env)
final_dqn.load_model('models/dqn_model.pth')

movie-recommender-rl/
│
├── 📓 movie_recommender_complete.ipynb  
├── 📄 movie_recommender_complete.py     
│
├── 📁 models/                           # 학습된 모델 저장
│   └── dqn_model.pth                    
│
├── 📁 data/                             # 데이터 (자동 다운로드)
│   └── ml-latest-small/
│       ├── ratings.csv
│       └── movies.csv
│
├── 📁 results/                          # 실험 결과
│   ├── final_results.csv
│   ├── epsilon_tuning.csv
│   └── dqn_tuning.csv
│
├── 📄 requirements.txt                  # 필요 라이브러리
├── 📄 README.md                         
└── 📄 project_report.ppt               # 프로젝트 보고서


## 📖 핵심 코드 설명

class MovieLensDataLoader:
    def __init__(self, save_dir='./data', min_user_ratings=20, min_movie_ratings=5):
        self.min_user_ratings = min_user_ratings  # 최소 20개 평점을 남긴 사용자만 포함
        self.min_movie_ratings = min_movie_ratings  # 최소 5개 평점을 받은 영화만 포함
- `min_user_ratings=20`: 평점을 20개 이상 남긴 사용자만 사용한다.
- `min_movie_ratings=5`: 평점을 5개 이상 받은 영화만 사용한다.

df['liked'] = (df['rating'] >= 4.0).astype(int)
- 5점 만점의 평점을 좋아요(1) / 싫어요(0)로 단순화한다.
- 4.0점 이상 → `liked = 1` (좋아함)
- 4.0점 미만 → `liked = 0` (좋아하지 않음)

def temporal_train_test_split(data, test_ratio=0.2):
    for user_id, user_df in data.groupby('userId'):
        user_df = user_df.sort_values('timestamp')  
        split_idx = int(len(user_df) * 0.8)
        train = user_df.iloc[:split_idx]   
        test = user_df.iloc[split_idx:]    
- 각 사용자별로 시간순으로 정렬.
- 과거 80%는 학습에, 미래 20%는 테스트에 사용한다.

def get_state(self, user_id):
    state = {
        'user_embed': self.user_embeddings[user_id],    
        'seq_embeds': seq_embeddings,                    
        'seq_rewards': seq_rewards                       
    }
    return state
- user_embed : SVD로 추출한 사용자의 취향 벡터. 
- seq_embeds : 최근에 본 영화의 벡터들. 최근 관심사를 반영.
- seq_rewards : 최근 본 영화를 좋아했는지(1) 싫어했는지(0).

def get_candidates(self, user_id, n=50):
    unwatched = self.all_movies - self.watched_movies[user_id]
    candidates = random.sample(list(unwatched), min(n, len(unwatched)))
    return candidates
- Action = 어떤 영화를 추천할지 선택
- 이때 각 step에서 전체 영화 중에서 50개 후보만 샘플링합니다.

def step(self, action):
    movie_id = action
    if movie_id in user_test_data:
        reward = 1 if user_liked_it else 0
    else:
        reward = 1 if svd_score > 0.5 else 0
    return next_state, reward, done, next_candidates
- Reward = 추천한 영화를 사용자가 좋아했는지
- 좋아했으면 1, 아니면 0

class RandomRecommender:
    def select_action(self, state, candidates):
        return random.choice(candidates)
- 가장 단순한 방법으로 아무 영화나 랜덤하게 추천.

class EpsilonGreedyRecommender:
    def select_action(self, state, candidates):
        if random.random() < self.epsilon:  
            return random.choice(candidates)  
        else:  
            scores = [(m, self.get_svd_score(user, m)) for m in candidates]
            return max(scores, key=lambda x: x[1])[0]
- 탐험 : 랜덤하게 선택해서 새로운 영화 발견
- 활용 : 지금까지 배운 지식으로 최선의 선택

class ThompsonSamplingRecommender:
    def __init__(self):
        self.alpha = defaultdict(lambda: 1)  
        self.beta = defaultdict(lambda: 1)   
    
    def select_action(self, state, candidates):
        samples = []
        for movie in candidates:
            theta = np.random.beta(self.alpha[movie], self.beta[movie])
            samples.append((movie, theta))
        return max(samples, key=lambda x: x[1])[0]
    
    def update(self, movie, reward):
        if reward == 1:
            self.alpha[movie] += 1  
        else:
            self.beta[movie] += 1   
- 각 영화가 좋아요를 받을 확률을 확률 분포로 모델링.
- 데이터가 많은 영화는 분포가 좁고 데이터가 적은 영화는 분포가 넓음

class SequentialDQNNetwork(nn.Module):
    def __init__(self, embed_dim=20, hidden_dim=128, lstm_hidden=64):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=embed_dim + 1,  
            hidden_size=lstm_hidden,    
            num_layers=2,              
            batch_first=True
        )
        
        input_dim = embed_dim + lstm_hidden + embed_dim  
        self.q_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),  
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)       
        )
- LSTM : 최근 본 영화를 분석해서 사용자 패턴 파악
- Q-Network : 사용자 정보 + 시퀀스 정보 + 추천할 영화 정보를 합쳐서 Q값 예측


def forward(self, user_embed, seq_embeds, seq_rewards, movie_embed):
    rewards_expanded = seq_rewards.unsqueeze(-1)  
    lstm_input = torch.cat([seq_embeds, rewards_expanded], dim=-1)  

    _, (h_n, _) = self.lstm(lstm_input)
    seq_encoding = h_n[-1]
    
    combined = torch.cat([user_embed, seq_encoding, movie_embed], dim=-1)  
    
    return self.q_network(combined)  
- 입력: 사용자 임베딩, 최근 본 영화들, 각 영화 좋아요 여부, 추천할 영화
- 출력: Q값

class SequentialReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action_id, movie_embed, reward, next_state, next_candidates, done):
        self.buffer.append((state, action_id, movie_embed, reward, 
                           next_state, next_candidates, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
- 경험을 저장해뒀다가 여러 번 학습에 재사용.
- 데이터 효율성이 높아지고, 학습이 안정적.

def _compute_max_next_q(self, next_user_embed, next_seq_embeds, next_seq_rewards, next_candidates):
    if len(next_candidates) == 0:
        return 0.0
    
    if len(next_candidates) > self.max_next_candidates:
        sampled = random.sample(next_candidates, self.max_next_candidates)
    else:
        sampled = next_candidates
    
    max_q = float('-inf')
    for movie_id in sampled:
        movie_embed = self.env.get_movie_embedding(movie_id)
        movie_embed = torch.FloatTensor(movie_embed).unsqueeze(0).to(device)
        
        q = self.target_net(next_user_embed, next_seq_embeds, next_seq_rewards, movie_embed)
        max_q = max(max_q, q.item())
    
    return max_q
- Q-learning : Target = reward + γ × max Q(s', a') = 지금 받은 보상 + 미래에 받을 수 있는 최대 보상

def train_step(self):
    batch = self.buffer.sample(self.batch_size)
    
    current_q = self.policy_net(
        batch['user_embeds'], batch['seq_embeds'], 
        batch['seq_rewards'], batch['movie_embeds']
    ).squeeze()
    
    with torch.no_grad():
        target_q_values = []
        for i in range(self.batch_size):
            reward = batch['rewards'][i].item()
            done = batch['dones'][i].item()
            
            if done:
                target_q = reward  
            else:
                max_next_q = self._compute_max_next_q(...)
                target_q = reward + self.gamma * max_next_q
            
            target_q_values.append(target_q)
    
    loss = F.smooth_l1_loss(current_q, torch.FloatTensor(target_q_values))
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
- 현재 Q값이 Target Q값에 가까워지도록 학습

def ndcg_at_k(self, user_id, recs, k):
    actual = self.user_ground_truth[user_id]
    
    dcg = sum(1.0 / np.log2(i + 2) for i, m in enumerate(recs[:k]) if m in actual)
    
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    
    return dcg / idcg if idcg > 0 else 0.0
- 상위에 있을수록 점수를 더 얻음.

def save_model(self, path):
    torch.save({
        'policy_net': self.policy_net.state_dict(),
        'target_net': self.target_net.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'epsilon': self.epsilon
    }, path)
    print(f"모델 저장 완료: {path}")
def load_model(self, path):
    checkpoint = torch.load(path)
    self.policy_net.load_state_dict(checkpoint['policy_net'])
    self.target_net.load_state_dict(checkpoint['target_net'])
    self.optimizer.load_state_dict(checkpoint['optimizer'])
    self.epsilon = checkpoint['epsilon']
    print(f"모델 로드 완료: {path}")

## 학습된 모델

`models/` 폴더에 학습된 DQN 모델이 저장되어 있습니다.
models/
└── dqn_model.pth
```
### 모델 사용 방법
# 1. 환경 및 추천기 초기화
env = RecommendationEnv(train_data, test_data, ...)
recommender = SequentialDQNRecommender(env)

# 2. 저장된 모델 로드
recommender.load_model('models/dqn_model.pth')

# 3. 추천 받기 (학습 없이 바로 사용 가능)
recommendations = recommender.get_recommendations(user_id=42, k=10)
for rec in recommendations:
    print(f"영화 ID: {rec['movieId']}, Q값: {rec['score']:.4f}")
