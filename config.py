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
TRAIN_EVERY = 2
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
