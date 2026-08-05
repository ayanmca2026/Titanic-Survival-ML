# Model Explainability Report (XAI)

## 1. Global Explanations (SHAP Summary)
*Which features impact the model the most overall?*
- **Sex_male**: Strongest negative impact on survival.
- **Pclass**: Lower class (3) negatively impacts survival.
- **Age**: Younger passengers have higher survival probability.
- **Fare**: Higher fare correlates with higher survival.

## 2. Local Explanations (LIME / SHAP)
*Example of a single prediction explanation:*
- Passenger A (Female, Class 1, Age 25): High probability of survival driven by Gender (Female) and Class (1).
- Passenger B (Male, Class 3, Age 30): Low probability of survival driven by Gender (Male) and Class (3).

## 3. Conclusion
The model relies heavily on socio-demographic features (Sex, Pclass, Age) aligned with the historical "Women and children first" maritime protocol.
