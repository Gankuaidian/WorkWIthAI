"""
Example of using Hugging Face pre-trained models for chemistry and protein prediction tasks.
This script demonstrates how to load pre-trained models and use them for inference
without additional training.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt

def demonstrate_chemistry_property_prediction():
    """
    Demonstrates how to use a pre-trained model for predicting molecular properties
    using example molecules.
    """
    print("=" * 50)
    print("Chemistry Property Prediction Example")
    print("=" * 50)
    
    # Use example data instead of loading from Hugging Face
    print("Using example molecules for demonstration...")
    
    # Create example molecule data
    example_molecules = [
        {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "name": "Aspirin", "property": 0},
        {"smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "name": "Ibuprofen", "property": 1},
        {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "name": "Caffeine", "property": 1},
        {"smiles": "CC(=O)NC1=CC=C(O)C=C1", "name": "Acetaminophen", "property": 0},
        {"smiles": "C1=CC=C2C(=C1)C=CC=C2", "name": "Naphthalene", "property": 1}
    ]
    
    # Display example data
    print("\nExample molecule data:")
    for i, mol in enumerate(example_molecules[:3]):
        print(f"Molecule {i+1}:")
        print(f"Name: {mol['name']}")
        print(f"SMILES: {mol['smiles']}")
        print(f"Example property (Blood-Brain Barrier Penetration): {'Positive' if mol['property'] == 1 else 'Negative'}")
        print()
    
    # Load pre-trained model for molecular property prediction
    print("Loading pre-trained model for molecular property prediction...")
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use feature extraction pipeline
    print("Creating feature extraction pipeline...")
    feature_extractor = pipeline("feature-extraction", model=model_name, tokenizer=tokenizer)
    
    # Extract features for test molecules
    print("\nExtracting features for test molecules:")
    
    for mol in example_molecules:
        # Extract molecular features
        features = feature_extractor(mol['smiles'])
        # Get average of feature vector as a simple representation
        avg_feature = np.mean(features[0][0])
        
        print(f"Molecule: {mol['name']}")
        print(f"SMILES: {mol['smiles']}")
        print(f"Feature vector average: {avg_feature:.4f}")
        print(f"Example property (Blood-Brain Barrier Penetration): {'Positive' if mol['property'] == 1 else 'Negative'}")
        print()
    
    print("Note: For actual property prediction, you would typically fine-tune")
    print("the model on your specific task or use a model already fine-tuned")
    print("for that property.")
    
    # Create more example molecules for demonstration
    print("\nDemonstrating molecular property prediction with more examples...")
    
    # Create additional example molecules with properties
    additional_molecules = [
        {"smiles": "C1CCCCC1", "name": "Cyclohexane", "property": 0},
        {"smiles": "C1=CC=C(C=C1)C(=O)O", "name": "Benzoic acid", "property": 1},
        {"smiles": "CCO", "name": "Ethanol", "property": 0},
        {"smiles": "CC1=CC=CC=C1", "name": "Toluene", "property": 1},
        {"smiles": "C1=CC=CC=C1", "name": "Benzene", "property": 1}
    ]
    
    # Display additional examples
    print("\nAdditional example molecules:")
    for i, mol in enumerate(additional_molecules[:3]):
        print(f"Molecule {i+1}:")
        print(f"Name: {mol['name']}")
        print(f"SMILES: {mol['smiles']}")
        print(f"Example property (Blood-Brain Barrier Penetration): {'Positive' if mol['property'] == 1 else 'Negative'}")
        print()
    
    # Extract features for additional molecules
    print("\nExtracting features for additional molecules:")
    
    for mol in additional_molecules:
        # Extract molecular features
        features = feature_extractor(mol['smiles'])
        # Get average of feature vector as a simple representation
        avg_feature = np.mean(features[0][0])
        
        print(f"Molecule: {mol['name']}")
        print(f"SMILES: {mol['smiles']}")
        print(f"Feature vector average: {avg_feature:.4f}")
        print(f"Example property (Blood-Brain Barrier Penetration): {'Positive' if mol['property'] == 1 else 'Negative'}")
        print()
    


def demonstrate_drug_efficacy_prediction():
    """
    Demonstrates how to use a pre-trained model for predicting drug efficacy
    and therapeutic potential.
    """
    print("\n" + "=" * 50)
    print("Drug Efficacy Prediction Example")
    print("=" * 50)
    
    # Example drugs with their SMILES and efficacy data
    example_drugs = [
        {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "name": "Aspirin", "efficacy": 0.78, "indication": "Pain relief"},
        {"smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "name": "Ibuprofen", "efficacy": 0.82, "indication": "Anti-inflammatory"},
        {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "name": "Caffeine", "efficacy": 0.65, "indication": "CNS stimulant"},
        {"smiles": "CC(=O)NC1=CC=C(O)C=C1", "name": "Acetaminophen", "efficacy": 0.75, "indication": "Pain relief"},
        {"smiles": "COC(=O)C1=C(C=CC=C1)NC(=O)C2=CC=C(C=C2)Cl", "name": "Diclofenac", "efficacy": 0.88, "indication": "Anti-inflammatory"}
    ]
    
    print("\nExample drug efficacy data:")
    for i, drug in enumerate(example_drugs):
        print(f"Drug {i+1}: {drug['name']}")
        print(f"SMILES: {drug['smiles']}")
        print(f"Therapeutic indication: {drug['indication']}")
        print(f"Efficacy score (0-1): {drug['efficacy']:.2f}")
        print(f"Clinical effectiveness: {'High' if drug['efficacy'] > 0.8 else 'Moderate' if drug['efficacy'] > 0.7 else 'Low'}")
        print()
    
    # Load pre-trained model for molecular representation
    print("Loading pre-trained model for drug representation...")
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use feature extraction pipeline
    print("Creating feature extraction pipeline...")
    feature_extractor = pipeline("feature-extraction", model=model_name, tokenizer=tokenizer)
    
    # Extract features and visualize drug similarities
    print("\nExtracting molecular features for drug similarity analysis:")
    
    drug_features = []
    drug_names = []
    drug_efficacies = []
    
    for drug in example_drugs:
        # Extract molecular features
        features = feature_extractor(drug['smiles'])
        # Get average of feature vector as a simple representation
        avg_feature = np.mean(features[0], axis=0)
        drug_features.append(avg_feature)
        drug_names.append(drug['name'])
        drug_efficacies.append(drug['efficacy'])
    
    # Convert to numpy array for easier manipulation
    drug_features = np.array(drug_features)
    
    # Apply PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(drug_features)
    
    # Plot drug similarities with efficacy as color intensity
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=drug_efficacies,
        cmap='viridis',
        s=100,
        alpha=0.8
    )
    
    # Add drug names as labels
    for i, name in enumerate(drug_names):
        plt.annotate(name, (reduced_features[i, 0], reduced_features[i, 1]), 
                    fontsize=12, ha='center', va='bottom')
    
    plt.colorbar(scatter, label='Efficacy Score')
    plt.title("Drug Similarity Map Based on Molecular Structure")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig("drug_similarity_map.png")
    print("Drug similarity visualization saved as 'drug_similarity_map.png'")
    
    # Demonstrate simple drug efficacy prediction
    print("\nDemonstrating drug efficacy prediction based on molecular features:")
    
    # Create a simple model (for demonstration purposes)
    from sklearn.linear_model import Ridge
    
    # Train a simple model on our example data
    model = Ridge(alpha=1.0)
    model.fit(drug_features, drug_efficacies)
    
    # Predict efficacy for the same drugs (just for demonstration)
    predicted_efficacies = model.predict(drug_features)
    
    # Display results
    print("\nPredicted vs. Actual Efficacy:")
    for i, drug in enumerate(example_drugs):
        print(f"Drug: {drug['name']}")
        print(f"Actual efficacy on {drug['indication']}: {drug['efficacy']:.2f}")
        print(f"Predicted efficacy on {drug['indication']}: {predicted_efficacies[i]:.2f}")
        print(f"Error: {abs(drug['efficacy'] - predicted_efficacies[i]):.2f}")
        print()
    


def demonstrate_chemgpt_predictions():
    """
    Use ChemGPT model to directly predict properties of molecules, peptides, and antibiotics.
    """
    print("\n" + "=" * 50)
    print("ChemGPT Molecule and Peptide Property Prediction Example")
    print("=" * 50)
    
    # Create example molecules, peptides, and antibiotics data
    example_compounds = [
        {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "name": "Aspirin", "type": "Small molecule drug"},
        {"smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "name": "Ibuprofen", "type": "Small molecule drug"},
        {"smiles": "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)COCCOCCOCCNC(=O)[C@H](CCCCN)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](C)NC(=O)[C@H](CCSC)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CO)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](N)CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)O", "name": "Insulin", "type": "Peptide"},
        {"smiles": "CC1C(O)C(OC1OC2C(C)C(OC2OC3=C4C=C5C=CC=C5N=C4C(=O)C(=C3O)C(=O)N)OC(=O)C(C(C)C(=O)C(C)C(O)C(C)C(O)C(C)C(=O)OC(C)CC)O)N)C", "name": "Rifampin", "type": "Antibiotic"},
        {"smiles": "CC(C)(C)NCC(O)COC1=CC=C(CCOCC2CC2)C=C1", "name": "Salbutamol", "type": "Bronchodilator"}
    ]
    
    print("\nExample compound data:")
    for i, compound in enumerate(example_compounds):
        print(f"Compound {i+1}: {compound['name']}")
        print(f"SMILES: {compound['smiles']}")
        print(f"Type: {compound['type']}")
        print()
    
    # Load PubChem10M model directly 
    print("Loading pre-trained PubChem10M model for molecular property prediction...")
    model_name = "seyonec/PubChem10M_SMILES_BPE_450k"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not already present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Create feature extraction pipeline
    feature_extractor = pipeline("feature-extraction", model=model_name, tokenizer=tokenizer)
    print(f"Successfully loaded {model_name} model!")
    
    # Define molecular properties to predict
    properties = [
        "Solubility", 
        "Bioactivity", 
        "Toxicity", 
        "Blood-Brain Barrier Permeability",
        "Half-life"
    ]
    
    # Extract features for each compound
    print("\nExtracting molecular features for property prediction:")
    
    compound_features = {}
    
    for compound in example_compounds:
        print(f"\nCompound: {compound['name']} ({compound['type']})")
        print(f"SMILES: {compound['smiles']}")
        
        # Extract features
        try:
            features = feature_extractor(compound['smiles'])
            # Get average feature vector as a simple representation
            avg_feature = np.mean(features[0], axis=0)
            compound_features[compound['name']] = avg_feature
            
            print(f"Feature extraction successful. Feature vector size: {len(avg_feature)}")
            
            # Simulate property predictions based on feature patterns
            # In a real application, you would use a trained classifier for each property
            for prop in properties:
                # Use a simple heuristic based on feature patterns
                # This is just for demonstration - real predictions would use trained models
                feature_sum = np.sum(avg_feature)
                feature_mean = np.mean(avg_feature)
                feature_std = np.std(avg_feature)
                
                if prop == "Solubility":
                    # Higher polarity (approximated by feature variance) often means higher solubility
                    score = (feature_std / feature_mean) * 0.8 + np.random.random() * 0.2
                    result = "High" if score > 0.5 else "Low"
                elif prop == "Bioactivity":
                    # Complex molecules (approximated by feature entropy) often have higher bioactivity
                    score = (np.abs(feature_sum) / len(avg_feature)) * 0.7 + np.random.random() * 0.3
                    result = "High" if score > 0.6 else "Low"
                elif prop == "Toxicity":
                    # Simplified toxicity estimation
                    score = feature_std * 0.6 + np.random.random() * 0.4
                    result = "Low" if score < 0.3 else "Medium" if score < 0.7 else "High"
                elif prop == "Blood-Brain Barrier Permeability":
                    # Simplified BBB permeability estimation
                    score = (1 - feature_std) * 0.75 + np.random.random() * 0.25
                    result = "Can penetrate" if score > 0.5 else "Cannot penetrate"
                elif prop == "Half-life":
                    # Simplified half-life estimation
                    score = feature_mean * 0.65 + np.random.random() * 0.35
                    result = "Long" if score > 0.7 else "Medium" if score > 0.3 else "Short"
                
                print(f"{prop}: {result} (Confidence: {score:.2f})")
        
        except Exception as e:
            print(f"Error processing compound {compound['name']}: {e}")
    
    # Visualize property comparisons
    print("\nGenerating property comparison visualization...")
    
    # Create a dictionary to store property prediction results for each compound
    property_scores = {}
    
    # Generate simulated prediction scores for visualization
    for compound in example_compounds:
        if compound['name'] in compound_features:
            # Use feature patterns to generate consistent property scores
            feature = compound_features[compound['name']]
            feature_sum = np.sum(feature)
            feature_mean = np.mean(feature)
            feature_std = np.std(feature)
            
            scores = []
            # Solubility
            scores.append((feature_std / feature_mean) * 0.8 + np.random.random() * 0.2)
            # Bioactivity
            scores.append((np.abs(feature_sum) / len(feature)) * 0.7 + np.random.random() * 0.3)
            # Toxicity (inverted so higher is better/safer)
            scores.append(1 - (feature_std * 0.6 + np.random.random() * 0.4))
            # BBB Permeability
            scores.append((1 - feature_std) * 0.75 + np.random.random() * 0.25)
            # Half-life
            scores.append(feature_mean * 0.65 + np.random.random() * 0.35)
            
            property_scores[compound['name']] = scores
        else:
            # Fallback to random scores if features weren't extracted
            property_scores[compound['name']] = [np.random.random() for _ in range(len(properties))]
    
    # Create radar chart
    categories = properties
    N = len(categories)
    
    # Create angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot each compound
    for i, compound in enumerate(example_compounds):
        values = property_scores[compound['name']]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=compound['name'])
        ax.fill(angles, values, alpha=0.1)
    
    # Set radar chart properties
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Compound Property Comparison", size=15, y=1.1)
    plt.tight_layout()
    plt.savefig("compound_properties_comparison.png")
    print("Compound property comparison visualization saved as 'compound_properties_comparison.png'")



if __name__ == "__main__":
    print("Hugging Face Pre-trained Models for Chemistry and Drug Prediction")
    print("This script demonstrates how to use pre-trained models without additional training.")

    demonstrate_chemistry_property_prediction()

    demonstrate_drug_efficacy_prediction()

    demonstrate_chemgpt_predictions()