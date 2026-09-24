GRID = 8
ENVS = 256
PLAYERS = 4
STACK = 1
FRAME_SKIP = 1

GAMMA = 0.99
NSTEP = 5
LR = 3e-4
BATCH = 512
REPLAY_CAP = 200000
# 每几帧做一次梯度更新。这个数直接决定"每秒真正更新多少次参数"，也就是学习速度：
#   更新次数/秒 =（步/秒 ÷ 每帧转移数）÷ TRAIN_EVERY
# 早先设 2 是因为环境单进程跑不快、采样一帧要 30 毫秒以上，每帧都训会让采样等训练；
# 现在环境拆进多进程池、采样一帧只要 23 毫秒，而一次更新（batch=512）约 8.5 毫秒，
# 每帧都训能把 GPU 一直喂饱，更新频率直接从 18 次/秒翻到 32 次/秒左右——
# 对"同样的墙钟时间学得更多"来说，这个收益比单纯堆步数大得多。
TRAIN_EVERY = 1
TARGET_EVERY = 20000
LEARN_START = 20000

STEP_PENALTY = -0.001
ITEM_REWARD = 0.05

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 30000000
OPP_EPS = 0.08

POOL_SNAPSHOT_EVERY = 500000
POOL_MAX = 12
PFSP_POWER = 2.0

HIDDEN = 512
CONV_CH = (16, 32, 32)

# 评测间隔。评测本身是单进程跑的，一次要 30 秒上下；它是按步数触发的，
# 所以训练一快，同样的步数间隔对应的墙钟时间就变短、评测占的时间比例就变大——
# 提速到 7 万步/秒后，300 万步只有 43 秒，评测能吃掉接近一半的训练时间。
# 提到 1000 万步（约 2.5 分钟一次）就能把这份开销压到 15% 以内。
EVAL_EVERY = 10000000
EVAL_ENVS = 24
EVAL_FRAMES = 700

CHECKPOINT = "tank_checkpoint.pth"
STATE = "tank_state.pth"
LOG = "tank_train.log"
MONITOR_PORT = 8080
SEED = 1
