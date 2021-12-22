DOMAINS = [
    'walker',
    'quadruped',
    'jaco',
    'acrobot'
]

ACROBOT_TASKS = ['acrobot_balance']

WALKER_TASKS = [
    'walker_stand',
    'walker_walk',
    'walker_run',
    'walker_flip',
]

QUADRUPED_TASKS = [
    'quadruped_walk',
    'quadruped_run',
    'quadruped_stand',
    'quadruped_jump',
]

JACO_TASKS = [
    'jaco_reach_top_left',
    'jaco_reach_top_right',
    'jaco_reach_bottom_left',
    'jaco_reach_bottom_right',
]

TASKS = WALKER_TASKS + QUADRUPED_TASKS + JACO_TASKS + ACROBOT_TASKS

PRIMAL_TASKS = {
    'acrobot': 'acrobot_balance'
    'walker': 'walker_stand',
    'jaco': 'jaco_reach_top_left',
    'quadruped': 'quadruped_walk'
}
