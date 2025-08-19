# ===============================
# Individual Model Training
# ===============================
# Copy this code into your Jupyter notebook as new cells

# Train Random Forest individually
print("🌳 Training Random Forest Model...")
print("=" * 50)

# Start timing
start_time = time()

# Create and train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_res, y_train_res)

# Calculate training time
rf_training_time = time() - start_time
print(f"✅ Random Forest trained in {rf_training_time:.4f} seconds")

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test_prep)
y_proba_rf = rf_model.predict_proba(X_test_prep)[:, 1]

# Calculate metrics
rf_acc = accuracy_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf)
rf_rec = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_f2 = fbeta_score(y_test, y_pred_rf, beta=2)
rf_roc_auc = roc_auc_score(y_test, y_proba_rf)

print(f"\n📊 Random Forest Performance:")
print(f"Accuracy: {rf_acc:.4f}")
print(f"Precision: {rf_prec:.4f}")
print(f"Recall: {rf_rec:.4f}")
print(f"F1 Score: {rf_f1:.4f}")
print(f"F2 Score: {rf_f2:.4f}")
print(f"ROC AUC: {rf_roc_auc:.4f}")

# Show classification report
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.show()

# Plot ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkgreen', lw=2, 
         label=f'Random Forest ROC (AUC = {roc_auc_rf:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc="lower right")
plt.show()

print("=" * 50)
print("✅ Random Forest training and evaluation complete!\n")

# ===============================
# Train SVM individually
print("🔷 Training SVM Model...")
print("=" * 50)

# Start timing
start_time = time()

# Create and train SVM
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_res, y_train_res)

# Calculate training time
svm_training_time = time() - start_time
print(f"✅ SVM trained in {svm_training_time:.4f} seconds")

# Evaluate SVM
y_pred_svm = svm_model.predict(X_test_prep)
y_proba_svm = svm_model.predict_proba(X_test_prep)[:, 1]

# Calculate metrics
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_prec = precision_score(y_test, y_pred_svm)
svm_rec = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)
svm_f2 = fbeta_score(y_test, y_pred_svm, beta=2)
svm_roc_auc = roc_auc_score(y_test, y_proba_svm)

print(f"\n📊 SVM Performance:")
print(f"Accuracy: {svm_acc:.4f}")
print(f"Precision: {svm_prec:.4f}")
print(f"Recall: {svm_rec:.4f}")
print(f"F1 Score: {svm_f1:.4f}")
print(f"F2 Score: {svm_f2:.4f}")
print(f"ROC AUC: {svm_roc_auc:.4f}")

# Show classification report
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.show()

# Plot ROC curve
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure()
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, 
         label=f'SVM ROC (AUC = {roc_auc_svm:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend(loc="lower right")
plt.show()

print("=" * 50)
print("✅ SVM training and evaluation complete!\n")

# ===============================
# Train ANN individually
print("🧠 Training ANN (Multi-layer Perceptron) Model...")
print("=" * 50)

# Start timing
start_time = time()

# Create and train ANN
ann_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
ann_model.fit(X_train_res, y_train_res)

# Calculate training time
ann_training_time = time() - start_time
print(f"✅ ANN trained in {ann_training_time:.4f} seconds")

# Evaluate ANN
y_pred_ann = ann_model.predict(X_test_prep)
y_proba_ann = ann_model.predict_proba(X_test_prep)[:, 1]

# Calculate metrics
ann_acc = accuracy_score(y_test, y_pred_ann)
ann_prec = precision_score(y_test, y_pred_ann)
ann_rec = recall_score(y_test, y_pred_ann)
ann_f1 = f1_score(y_test, y_pred_ann)
ann_f2 = fbeta_score(y_test, y_pred_ann, beta=2)
ann_roc_auc = roc_auc_score(y_test, y_proba_ann)

print(f"\n📊 ANN Performance:")
print(f"Accuracy: {ann_acc:.4f}")
print(f"Precision: {ann_prec:.4f}")
print(f"Recall: {ann_rec:.4f}")
print(f"F1 Score: {ann_f1:.4f}")
print(f"F2 Score: {ann_f2:.4f}")
print(f"ROC AUC: {ann_roc_auc:.4f}")

# Show classification report
print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_ann))

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_ann, cmap='Blues')
plt.title("ANN Confusion Matrix")
plt.show()

# Plot ROC curve
fpr_ann, tpr_ann, _ = roc_curve(y_test, y_proba_ann)
roc_auc_ann = auc(fpr_ann, tpr_ann)

plt.figure()
plt.plot(fpr_ann, tpr_ann, color='purple', lw=2, 
         label=f'ANN ROC (AUC = {roc_auc_ann:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ANN ROC Curve')
plt.legend(loc="lower right")
plt.show()

print("=" * 50)
print("✅ ANN training and evaluation complete!\n")

# ===============================
# Individual Model Comparison
print("📈 Individual Model Comparison Summary")
print("=" * 60)

# Create comparison DataFrame
individual_results = pd.DataFrame({
    'Model': ['Random Forest', 'SVM', 'ANN'],
    'Training Time (s)': [rf_training_time, svm_training_time, ann_training_time],
    'Accuracy': [rf_acc, svm_acc, ann_acc],
    'Precision': [rf_prec, svm_prec, ann_prec],
    'Recall': [rf_rec, svm_rec, ann_rec],
    'F1 Score': [rf_f1, svm_f1, ann_f1],
    'F2 Score': [rf_f2, svm_f2, ann_f2],
    'ROC AUC': [rf_roc_auc, svm_roc_auc, ann_roc_auc]
})

# Round numeric columns
numeric_cols = ['Training Time (s)', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'F2 Score', 'ROC AUC']
individual_results[numeric_cols] = individual_results[numeric_cols].round(4)

# Display comparison table
print(individual_results.to_string(index=False))

# Find best model for each metric
print(f"\n🏆 Best Model by Metric:")
print(f"Accuracy: {individual_results.loc[individual_results['Accuracy'].idxmax(), 'Model']}")
print(f"Precision: {individual_results.loc[individual_results['Precision'].idxmax(), 'Model']}")
print(f"Recall: {individual_results.loc[individual_results['Recall'].idxmax(), 'Model']}")
print(f"F1 Score: {individual_results.loc[individual_results['F1 Score'].idxmax(), 'Model']}")
print(f"F2 Score: {individual_results.loc[individual_results['F2 Score'].idxmax(), 'Model']}")
print(f"ROC AUC: {individual_results.loc[individual_results['ROC AUC'].idxmax(), 'Model']}")
print(f"Fastest Training: {individual_results.loc[individual_results['Training Time (s)'].idxmin(), 'Model']}")

# Store individual models for later use
individual_models = {
    'Random Forest': rf_model,
    'SVM': svm_model,
    'ANN': ann_model
}

print("\n✅ Individual model training complete! Models stored in 'individual_models' dictionary.")
print("You can now use individual_models['Model Name'] to access specific models.")
