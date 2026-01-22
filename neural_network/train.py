from neural_network.model import *
import os


def train_contrastive_model(hdf5_path, dataset_key='data', batch_size=256, output_dim=32, num_epochs=500, device='cuda', 
                            save_path='contrastive_model.pth', log_dir=None, experiment_name=None, augmentation_config=None):
    """Main training function"""
    
    # Create dataset and dataloader
    dataset = HDF5Dataset(hdf5_path, dataset_key=dataset_key, augmentation=augmentation_config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = ContrastiveModel(backbone_feature_dim=128, output_dim=output_dim)
    
    if os.path.exists(os.path.dirname(save_path)):
        checkpoint = torch.load(os.path.join(os.path.dirname(save_path), "ckpt_best.pth"), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {os.path.join(os.path.dirname(save_path), 'ckpt_best.pth')}")
    
    learner = ContrastiveLearner(
        model, 
        device=device, 
        lr=1e-3,
        log_dir=log_dir,
        experiment_name=experiment_name
    )
    
    learner.writer.add_text('Model/Architecture', 'Single Channel CNN on Footprint Only')
    learner.writer.add_text('Model/Total_Parameters', f'{370880:,} parameters')
    learner.writer.add_text('Training/Batch_Size', str(batch_size))
    learner.writer.add_text('Training/Learning_Rate', str(1e-3))
    learner.writer.add_text('Training/Dataset_Size', f'{len(dataset):,} samples')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    try:
        for epoch in range(num_epochs):
            # Train epoch
            avg_loss = learner.train_epoch(dataloader, epoch)
            learner.scheduler.step()
            
            # Print progress
            current_lr = learner.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                learner.save_model(os.path.join(save_path, "ckpt_best.pth"))
                learner.writer.add_scalar('Loss/Best', best_loss, epoch)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_path, f"ckpt_epoch_{epoch+1}.pth")
                learner.save_model(checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
            
            # Log validation metrics
            # learner.writer.add_scalar('Loss/Validation', val_loss, epoch)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        learner.save_model(os.path.join(save_path, f"ckpt_epoch_{epoch+1}.pth"))
        print(f"Training completed! Final model saved to {os.path.join(save_path, f'ckpt_epoch_{epoch+1}.pth')}")
        print(f"Best model saved to {os.path.join(save_path, 'ckpt_best.pth')} (loss: {best_loss:.4f})")

        learner.close_tensorboard()
    
    return learner


if __name__ == "__main__":

    hdf5_path = r"\\path\to\dataset.h5"

    exp_name = "footprint_only_contrastive_16d_augmented"

    augmentations = Augmentation3D()                # default augmentation settings
    
    learner = train_contrastive_model(
        hdf5_path=hdf5_path,
        dataset_key='data',
        batch_size=256,
        output_dim=16,
        num_epochs=300,
        device='cuda',
        save_path=os.path.join(r"path\to\save\models", exp_name, "ckpt"),
        log_dir=os.path.join(r"path\to\save\models", exp_name, "logs"),
        experiment_name=exp_name,
        augmentation_config=augmentations
    )
    
    # Extract trained model embeddings
    dataset = HDF5Dataset(hdf5_path, dataset_key='data')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    
    features = learner.extract_features(dataloader)
    print(f"Extracted features shape: {features.shape}")
    
    # Save features for later use
    np.save(f'{exp_name}_features.npy', features)
    print(f"Features saved to {exp_name}_features.npy")
