def evaluate_model(model, X_val, y_val, val_preds, feature_names, reverse_map, points_val=None, title_prefix="Model"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np

    # 映射回原始标签
    y_val_original = np.array([reverse_map[int(y)] for y in y_val])
    val_preds_original = np.array([reverse_map[int(p)] for p in val_preds])

    # 唯一标签
    unique_labels = sorted(set(np.unique(y_val_original)) | set(np.unique(val_preds_original)))
    label_names = {0:'Other', 1:'Sea Surface', 2:'Seabed', 4:'Water Body'}
    target_names = [f"{label_names[label]} ({label})" for label in unique_labels]

    # --- 1. 分类报告 ---
    print(f"\n=== {title_prefix} Classification Report ===")
    print(classification_report(y_val_original, val_preds_original,
                                labels=unique_labels,
                                target_names=target_names))
    
    # --- 2. 混淆矩阵 ---
    cm = confusion_matrix(y_val_original, val_preds_original, labels=unique_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10,8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.title(f'{title_prefix} Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, target_names, rotation=45, ha='right')
    plt.yticks(tick_marks, target_names)
    
    thresh = cm_normalized.max() / 2.
    for i, j in np.ndindex(cm_normalized.shape):
        plt.text(j, i, f'{cm_normalized[i,j]:.2%}',
                 horizontalalignment='center',
                 color='white' if cm_normalized[i,j] > thresh else 'black')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

    # --- 3. 特征重要性 ---
    importance_scores = model.get_score(importance_type='gain')
    importance_dict = {feature_names[int(k[1:])]: v for k,v in importance_scores.items()}
    sorted_features = sorted(importance_dict.items(), key=lambda x:x[1], reverse=True)

    plt.figure(figsize=(12,6))
    feats, scores = zip(*sorted_features)
    plt.bar(feats, scores)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{title_prefix} Feature Importance')
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== {title_prefix} Feature Importance Ranking ===")
    for feat, score in sorted_features:
        print(f"{feat:20s}: {score:.4f}")

    # --- 4. 特征相关性 ---
    plt.figure(figsize=(12,10))
    corr_matrix = np.corrcoef(X_val.T)
    plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title(f'{title_prefix} Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # --- 5. T-Z 投影图（如果传入了坐标 points_val） ---
    if points_val is not None:
        plt.figure(figsize=(15,5))
        # Ground Truth
        plt.subplot(121)
        scatter = plt.scatter(points_val[:,0], points_val[:,2], c=y_val_original, 
                             cmap='tab10', s=1, alpha=0.6)
        plt.colorbar(scatter, ticks=unique_labels)
        plt.title(f'{title_prefix} Ground Truth (T-Z)')
        plt.xlabel('Time')
        plt.ylabel('Depth')
        # Predictions
        plt.subplot(122)
        scatter = plt.scatter(points_val[:,0], points_val[:,2], c=val_preds_original,
                             cmap='tab10', s=1, alpha=0.6)
        plt.colorbar(scatter, ticks=unique_labels)
        plt.title(f'{title_prefix} Predictions (T-Z)')
        plt.xlabel('Time')
        plt.ylabel('Depth')
        plt.tight_layout()
        plt.show()
