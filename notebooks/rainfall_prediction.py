{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b0ad6349-fc88-4d62-8afa-e882ad196934",
   "metadata": {},
   "outputs": [],
   "source": [
    "import feature_engineering as fe\n",
    "\n",
    "train = fe.train\n",
    "test = fe.test\n",
    "\n",
    "model = XGBClassifier(\n",
    "    device=\"cuda\",\n",
    "    max_depth=6,\n",
    "    colsample_bytree=0.9,\n",
    "    subsample=0.9,\n",
    "    n_estimators=10_000,\n",
    "    learning_rate=0.1,\n",
    "    eval_metric=\"auc\",\n",
    "    alpha=1,\n",
    ")\n",
    "\n",
    "x_train = train[FEATURES+ADD].copy()\n",
    "y_train = train[\"rainfall\"]\n",
    "x_test = test[FEATURES+ADD].copy()\n",
    "model.fit(x_train, y_train)\n",
    "\n",
    "# INFER OOF\n",
    "outputs = model.predict_proba(x_test)[:,1]\n",
    "\n",
    "df_submission = pd.read_csv('sample_submission.csv')\n",
    "df_submission['rainfall'] = outputs\n",
    "df_submission.to_csv('submission.csv', index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
