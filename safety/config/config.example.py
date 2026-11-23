# Copy to config/config.py and edit. Any SafetyConfig field can be set as SAFETY_<FIELD_NAME_UPPER>.
MODEL_PATH = "weights/best.pt"                  # YOLO model (inserted / loose / occluded)
BACKGROUND_PATH = "background_empty_tray.npz"   # captured with key 'b' on an empty tray
SAM_EVERY_N = 0                                 # 0 = SAM 3 only on key 's'
SAFETY_NUT_SIDE_MM = (17, 35)                   # use key 'm' to measure real sizes first
SAFETY_BG_MISMATCH_FRAC = 0.20
# SAFETY_CAM_TO_ROBOT = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]   # 4x4 hand-eye matrix
