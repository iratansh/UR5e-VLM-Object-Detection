using UnityEngine;
using System.IO;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class BehaviorCloningTrainer : MonoBehaviour
{
    public string demonstrationsPath = "Assets/Demonstrations";
    public string demonstrationFileName = "UR5e_Grasp_Demo.demo";
    public string outputModelPath = "Assets/Models";
    public string modelName = "UR5e_Grasping_Model";

    public bool trainOnStart = false;
    public int trainingSteps = 20000;

    private void Start()
    {
        if (trainOnStart)
        {
            StartBehaviorCloningTraining();
        }
    }

    public void StartBehaviorCloningTraining()
    {
#if UNITY_EDITOR

        string demoPath = System.IO.Path.Combine(demonstrationsPath, demonstrationFileName);
        string fullOutputPath = System.IO.Path.Combine(outputModelPath, modelName);

        Debug.Log($"To train using behavior cloning, run this command externally:");
        Debug.Log($"mlagents-learn --force " +
                  $"--trainer-config-path=config/grasping_config.yaml " +
                  $"--time-scale=1 " +
                  $"--env=./Build/YourBuildName " +
                  $"--run-id={modelName} " +
                  $"--num-envs=1 " +
                  $"--demo-path={demoPath} " +
                  $"--base-port=5005");

        Debug.Log("Make sure to create a mlagents-learn configuration file with behavior cloning settings:");
        Debug.Log("behaviors:\n" +
                  "  RobotGrasping:\n" +
                  "    trainer_type: ppo\n" +
                  "    hyperparameters:\n" +
                  "      batch_size: 128\n" +
                  "      buffer_size: 2048\n" +
                  "      learning_rate: 0.0003\n" +
                  "      beta: 0.005\n" +
                  "      epsilon: 0.2\n" +
                  "      lambd: 0.95\n" +
                  "      num_epoch: 3\n" +
                  "      learning_rate_schedule: linear\n" +
                  "    behavioral_cloning:\n" +
                  "      demo_path: " + demoPath + "\n" +
                  "      strength: 0.5\n" +
                  "      steps: 150000\n" +
                  "    network_settings:\n" +
                  "      normalize: true\n" +
                  "      hidden_units: 256\n" +
                  "      num_layers: 3\n" +
                  "      vis_encode_type: simple\n" +
                  "    reward_signals:\n" +
                  "      extrinsic:\n" +
                  "        gamma: 0.99\n" +
                  "        strength: 1.0\n" +
                  "      gail:\n" +
                  "        gamma: 0.99\n" +
                  "        strength: 0.5\n" +
                  "        demo_path: " + demoPath + "\n" +
                  "    max_steps: " + trainingSteps + "\n" +
                  "    time_horizon: 64\n" +
                  "    summary_freq: 10000");
#endif
    }
}