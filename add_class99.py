import pandas as pd
import numpy as np

def pred_99(preds):
    class_99_probs = np.ones(preds.shape[0])

    if "class_99" in preds.columns:
        preds = preds.drop("class_99", axis=1)

    for col in preds.columns:
        class_99_probs *= (1 - preds.loc[:, col])

    class_99_probs = pd.DataFrame(class_99_probs)
    class_99_probs.columns = ["class_99"]
    class_99_probs /= class_99_probs.mean()
    class_99_probs *= 0.14

    preds = pd.concat([preds, class_99_probs], axis=1)
    return preds.div(preds.sum(axis=1), axis=0)

if __name__ == "__main__":
    submission_name = "submission.csv"
    print("Reading in {}".format(submission_name))
    sub = pd.read_csv(submission_name, index_col="object_id")
    new_preds = pred_99(sub)
    print("Writing")
    new_preds.to_csv("submission99.csv", float_format="%g")
