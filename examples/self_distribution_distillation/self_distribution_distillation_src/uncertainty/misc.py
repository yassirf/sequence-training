from typing import List, Dict


def process_outputs(outputs, extra: List[Dict]):
    """
    If extra has 'teacher_predictions_lp' then access these.
    """
    check = 'teacher_predictions_lp' in extra[0]

    # Returns a list of predictions
    if not check: return outputs

    # If these do exist then get the number of predictions in each member
    num = extra[0]['teacher_predictions_lp'].size(1)

    return [
        ex['teacher_predictions_lp'][:, i]
        for ex in extra
        for i in range(num)
    ]