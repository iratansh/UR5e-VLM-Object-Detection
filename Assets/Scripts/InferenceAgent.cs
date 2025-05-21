using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class InferenceAgent : Agent
{
    public string[] jointNames;
    private Dictionary<string, ArticulationBody> jointMap = new Dictionary<string, ArticulationBody>();

    public override void Initialize()
    {
        foreach (string jointName in jointNames)
        {
            var joint = GameObject.Find(jointName)?.GetComponent<ArticulationBody>();
            if (joint != null)
            {
                jointMap[jointName] = joint;
            }
        }

        if (jointMap.Count == 0)
        {
            Debug.LogError("⚠️ No valid articulation joints found for Unitree arm!");
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int i = 0;
        foreach (var pair in jointMap)
        {
            var joint = pair.Value;
            float targetDelta = actions.ContinuousActions[i++] * 0.1f;
            var drive = joint.xDrive;
            drive.target += targetDelta;
            joint.xDrive = drive;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        foreach (var joint in jointMap.Values)
        {
            sensor.AddObservation(joint.jointPosition[0]);
        }
    }

}
