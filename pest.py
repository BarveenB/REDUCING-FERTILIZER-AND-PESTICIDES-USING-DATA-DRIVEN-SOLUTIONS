import streamlit as st
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt

# Configuration and Model Definition
class PestClassifier:
    def __init__(self):
        self.classes = [
            'rice leaf roller', 
            'rice leaf caterpillar', 
            'paddy stem maggot',
            'asiatic rice borer', 
            'yellow rice borer', 
            'rice gall midge',
            'Rice Stemfly', 
            'brown plant hopper', 
            'white backed plant hopper',
            'small brown plant hopper'
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = r"E:\soundarya\crop recommentation\pest detection audio\pest detection audio\shufflenet_model.pth"
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self):
        model = models.shufflenet_v2_x1_0(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(self.classes))
        
        try:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            img_t = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_t)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            
            return {
                'class': self.classes[predicted.item()],
                'confidence': probabilities[predicted.item()].item(),
                'all_predictions': {
                    cls_name: prob.item() 
                    for cls_name, prob in zip(self.classes, probabilities)
                }
            }
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return None

# Streamlit App
def main():
    st.set_page_config(page_title="Pest Classifier", layout="wide")
    st.title("🌾 Agricultural Pest Classifier")
    
    # Initialize classifier
    classifier = PestClassifier()
    
    if classifier.model is None:
        st.error("Failed to load model. Please check the model file path.")
        return

    # Sidebar
    st.sidebar.header("Model Information")
    st.sidebar.write(f"Model: ShuffleNet v2")
    st.sidebar.write(f"Classes: {len(classifier.classes)} pest types")

    # Main interface
    uploaded_file = st.file_uploader("Upload pest image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        # Temporary save
        temp_path = "temp_prediction_image.jpg"
        image.save(temp_path)
        
        if st.button("Classify Pest"):
            with st.spinner("Analyzing..."):
                result = classifier.predict(temp_path)
            
            if result:
                st.success(f"Prediction: **{result['class']}** (Confidence: {result['confidence']:.2f}%)")
                
                # Show all predictions
                st.subheader("Detailed Predictions:")
                for class_name, confidence in result['all_predictions'].items():
                    st.progress(int(confidence), text=f"{class_name}: {confidence:.2f}%")
                
                # Visualization
                fig, ax = plt.subplots()
                classes = list(result['all_predictions'].keys())
                confidences = list(result['all_predictions'].values())
                ax.barh(classes, confidences)
                ax.set_xlabel('Confidence (%)')
                ax.set_title('Prediction Confidence Levels')
                st.pyplot(fig)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Instructions
    st.markdown("""
    ### How to Use:
    1. Upload an image of a plant with pest damage
    2. Click the "Classify Pest" button
    3. View the prediction results

    ### Supported Pest Types:
    - Rice leaf roller
    - Rice leaf caterpillar
    - Paddy stem maggot
    - Asiatic rice borer
    - Yellow rice borer
    - Rice gall midge
    - Rice Stemfly
    - Brown plant hopper
    - White backed plant hopper
    - Small brown plant hopper
    """)

if __name__ == "__main__":
    main()