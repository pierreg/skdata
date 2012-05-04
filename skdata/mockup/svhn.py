# -- model
from awesome import AwesomeClassifier  # an awesome semi-supervised classifier
clf = AwesomeClassifier()

# -- dataset / task
from skdata import svhn

model_selection_tasks = svhn.task.CroppedDigitsStratifiedKFoldOnTrain(
    k=5,
    use_extra=True,  # (dataset specific) use extra training data, see http://ufldl.stanford.edu/housenumbers/
    rng=np.random.RandomState(42),
    )

for task in model_selection_tasks:
    X_trn = task['X_train']  # lazy array over random shuffle of train + extra
    y_trn = task.y_train
    clf.fit(X_trn, y_trn)

    X_trn = task.X_test
    y_true = task.X_test
    y_pred = clf.predict(X_trn)

    accuracy = metric(y_true, y_pred)


final_task = svhn.CroppedDigitsOfficialTask(use_extra=True)

X_trn = task.X_train
y_trn = task.y_train
clf.fit(X_trn, y_trn)

y_true = task.y_test
y_pred = clf.predict(X_tst)

final_acc = metric(y_true, y_pred)
