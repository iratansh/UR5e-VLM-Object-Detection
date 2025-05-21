using UnityEngine;
using Unity.MLAgents.Policies;
using Unity.Sentis;
using UnityEditor; 

public class UnitreeIntegrationHelper : MonoBehaviour
{
    public string modelPath = "Assets/Models/UR5e_Grasping_Model.onnx"; 
    public GameObject unitreeArm;
    public string[] jointNames = { "joint1", "joint2", "joint3", "joint4", "joint5", "joint6" };
    
    private ModelAsset modelAsset;
    private BehaviorParameters behaviorParams;
    private InferenceAgent inferenceAgent;

    private void Start()
    {
        modelAsset = (ModelAsset)AssetDatabase.LoadAssetAtPath(modelPath, typeof(ModelAsset));
        if (modelAsset == null)
        {
            Debug.LogError("🛑 Failed to load ML-Agents model from path: " + modelPath);
            return;
        }

        behaviorParams = unitreeArm.GetComponent<BehaviorParameters>();
        if (behaviorParams == null)
        {
            behaviorParams = unitreeArm.AddComponent<BehaviorParameters>();
        }
        behaviorParams.BehaviorType = BehaviorType.InferenceOnly;
        behaviorParams.BehaviorName = "UnitreeGrasping";
        
        // behaviorParams.SentisModel = modelAsset;
        var modelProperty = typeof(BehaviorParameters).GetProperty("Model");
        if (modelProperty != null)
        {
            modelProperty.SetValue(behaviorParams, modelAsset);
        }

        inferenceAgent = unitreeArm.GetComponent<InferenceAgent>();
        if (inferenceAgent == null)
        {
            inferenceAgent = unitreeArm.AddComponent<InferenceAgent>();
        }

        inferenceAgent.jointNames = jointNames;

        Debug.Log("✅ UnitreeIntegrationHelper initialized with model: " + modelPath);
    }
}