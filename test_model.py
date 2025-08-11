import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from main import DigitRecognitionModel
import os

class DigitTester:
    def __init__(self, model_path='./modelsdigit_recognition_model.h5'):
        self.model = DigitRecognitionModel()
        if os.path.exists(model_path):
            self.model.load_model(model_path)
        else:
            print(f"Model file {model_path} not found. Please train the model first.")
            
    def preprocess_custom_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        img = cv2.resize(img, (28, 28))
        
        if np.mean(img) > 127:
            img = 255 - img

        img = img.astype('float32') / 255.0
        
        img = img.reshape(28, 28, 1)
        
        return img
        
    def test_custom_image(self, image_path):
        """Test the model on a custom image"""
        try:
            processed_img = self.preprocess_custom_image(image_path)

            predicted_digit, confidence = self.model.predict_digit(processed_img)

            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            plt.imshow(original, cmap='gray')
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(processed_img.reshape(28, 28), cmap='gray')
            plt.title(f'Predicted: {predicted_digit} (Confidence: {confidence:.2f})')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print(f"Prediction: {predicted_digit}")
            print(f"Confidence: {confidence:.4f}")
            
            return predicted_digit, confidence
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None
            
    def test_mnist_samples(self, num_samples=10):
        """Test the model on random MNIST samples"""
        if self.model.x_test is None:
            self.model.load_and_preprocess_data()

        indices = np.random.choice(len(self.model.x_test), num_samples, replace=False)
        
        plt.figure(figsize=(15, 6))
        
        for i, idx in enumerate(indices):
            img = self.model.x_test[idx]
            true_label = np.argmax(self.model.y_test[idx])

            predicted_digit, confidence = self.model.predict_digit(img)

            plt.subplot(2, 5, i + 1)
            plt.imshow(img.reshape(28, 28), cmap='gray')
            color = 'green' if predicted_digit == true_label else 'red'
            plt.title(f'True: {true_label}, Pred: {predicted_digit}\\nConf: {confidence:.2f}', 
                     color=color, fontsize=10)
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
        
    def create_drawing_interface(self):
        print("Drawing interface would require additional GUI setup.")
        print("For now, you can:")
        print("1. Draw digits using external tools (Paint, etc.)")
        print("2. Save as image files")
        print("3. Use test_custom_image() to test them")
        
    def batch_test_directory(self, directory_path):
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return
            
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        results = []
        
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(directory_path, filename)
                print(f"Testing {filename}...")
                
                predicted_digit, confidence = self.test_custom_image(image_path)
                if predicted_digit is not None:
                    results.append({
                        'filename': filename,
                        'prediction': predicted_digit,
                        'confidence': confidence
                    })
                    
        if results:
            print("\\n=== Batch Test Results ===")
            for result in results:
                print(f"{result['filename']}: {result['prediction']} (conf: {result['confidence']:.3f})")
                
        return results

def interactive_test():
    print("=== Digit Recognition Model Tester ===")
    
    tester = DigitTester()
    
    while True:
        print("\\nChoose an option:")
        print("1. Test with custom image file")
        print("2. Test random MNIST samples")
        print("3. Batch test directory")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            image_path = input("Enter path to image file: ").strip()
            if os.path.exists(image_path):
                tester.test_custom_image(image_path)
            else:
                print("File not found!")
                
        elif choice == '2':
            num_samples = input("Number of samples to test (default 10): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else 10
            tester.test_mnist_samples(num_samples)
            
        elif choice == '3':
            directory = input("Enter directory path: ").strip()
            tester.batch_test_directory(directory)
            
        elif choice == '4':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    interactive_test()
