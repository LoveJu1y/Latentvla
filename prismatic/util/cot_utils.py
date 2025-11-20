import enum


class CotTag(enum.Enum):
    TASK = "TASK:"
    PLAN = "PLAN:"
    VISIBLE_OBJECTS = "VISIBLE OBJECTS:"
    SUBTASK_REASONING = "SUBTASK REASONING:"
    SUBTASK = "SUBTASK:"
    MOVE_REASONING = "MOVE REASONING:"
    MOVE = "MOVE:"
    GRIPPER_POSITION = "GRIPPER POSITION:"
    ACTION = "ACTION:"


def abbreviate_tag(tag: str):
    if len(tag) < 2:
        return "TH"  # Default abbreviation for thinking tokens
    return tag[0] + tag[-2]


def get_cot_tags_list():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.VISIBLE_OBJECTS.value,
        CotTag.SUBTASK_REASONING.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE_REASONING.value,
        CotTag.MOVE.value,
        CotTag.GRIPPER_POSITION.value,
        CotTag.ACTION.value,
    ]
def get_cot_tags_list_simple():
    return [
        CotTag.TASK.value,
        CotTag.PLAN.value,
        CotTag.SUBTASK.value,
        CotTag.MOVE.value,
    ]

def get_cot_database_keys():
    return {
        CotTag.TASK.value: "task",
        CotTag.PLAN.value: "plan",
        CotTag.VISIBLE_OBJECTS.value: "bboxes",
        CotTag.SUBTASK_REASONING.value: "subtask_reason",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE_REASONING.value: "move_reason",
        CotTag.MOVE.value: "move",
        CotTag.GRIPPER_POSITION.value: "gripper",
        CotTag.ACTION.value: "action",
    }
def get_cot_database_keys_simple():
    return {
        CotTag.TASK.value: "task",
        CotTag.PLAN.value: "plan",
        CotTag.SUBTASK.value: "subtask",
        CotTag.MOVE.value: "move",
    }

def get_stage_config(scheduled_stage: int):
    """根据训练阶段返回应该包含的CoT组件"""
    all_tags = get_cot_tags_list()[:-1]  # 排除ACTION
    if scheduled_stage == 0:
        return all_tags  # 完整CoT
    elif scheduled_stage >= len(all_tags):
        return []  # 只保留ACTION
    else:
        return all_tags[scheduled_stage:]  # 从指定位置开始保留


def get_thinking_token_replacement():
    """获取thinking token替代字符串"""
    return "<|start_of_thinking|><|thinking|><|thinking|><|thinking|><|thinking|><|thinking|><|end_of_thinking|>"
