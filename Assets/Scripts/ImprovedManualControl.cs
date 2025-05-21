using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Policies;

public class ImprovedManualControl : MonoBehaviour
{
    public GraspingAgent graspingAgent;

    public float jointControlSpeed = 0.05f;
    public float gripperSpeed = 0.005f;

    private ArticulationBody[] robotJoints;
    private ArticulationBody gripperJoint;
    private ArticulationBody leftFingerJoint;
    private ArticulationBody rightFingerJoint;


    private float maxJointValue = 3.14f;

    void Start()
    {
        if (graspingAgent == null)
        {
            graspingAgent = GetComponent<GraspingAgent>();
            if (graspingAgent == null)
            {
                Debug.LogError("No GraspingAgent component found!");
                enabled = false;
                return;
            }
        }
        graspingAgent.manualControl = true;


        Academy.Instance.AutomaticSteppingEnabled = false;

        BehaviorParameters behaviorParams = GetComponent<BehaviorParameters>();
        if (behaviorParams != null)
        {
            behaviorParams.BehaviorType = BehaviorType.HeuristicOnly;
        }

        robotJoints = graspingAgent.robotJoints;
        gripperJoint = graspingAgent.gripperJoint;
        leftFingerJoint = graspingAgent.leftFingerJoint;
        rightFingerJoint = graspingAgent.rightFingerJoint;


        if (robotJoints == null || robotJoints.Length == 0)
        {
            Debug.LogError("No robot joints found in the GraspingAgent!");
            enabled = false;
            return;
        }

        maxJointValue = graspingAgent.maxJointValue;

        Debug.Log("Manual control initialized with " + robotJoints.Length + " joints");
    }

    void Update()
    {
        DirectJointControl();

        if (Input.GetKeyDown(KeyCode.Z))
        {
            ResetSimulation();
        }
        Academy.Instance.EnvironmentStep();
    }

    void DirectJointControl()
    {
        // Joint 0 (shoulder) - Q/A keys
        if (Input.GetKey(KeyCode.Q))
            MoveJoint(0, jointControlSpeed);
        else if (Input.GetKey(KeyCode.A))
            MoveJoint(0, -jointControlSpeed);

        // Joint 1 (upper arm) - W/S keys
        if (Input.GetKey(KeyCode.W))
            MoveJoint(1, jointControlSpeed);
        else if (Input.GetKey(KeyCode.S))
            MoveJoint(1, -jointControlSpeed);

        // Joint 2 (forearm) - E/D keys
        if (Input.GetKey(KeyCode.E))
            MoveJoint(2, jointControlSpeed);
        else if (Input.GetKey(KeyCode.D))
            MoveJoint(2, -jointControlSpeed);

        // Joint 3 (wrist_1) - R/F keys
        if (Input.GetKey(KeyCode.R))
            MoveJoint(3, jointControlSpeed);
        else if (Input.GetKey(KeyCode.F))
            MoveJoint(3, -jointControlSpeed);

        // Joint 4 (wrist_2) - T/G keys
        if (Input.GetKey(KeyCode.T))
            MoveJoint(4, jointControlSpeed);
        else if (Input.GetKey(KeyCode.G))
            MoveJoint(4, -jointControlSpeed);

        // Joint 5 (wrist_3) - Y/H keys
        if (Input.GetKey(KeyCode.Y))
            MoveJoint(5, jointControlSpeed);
        else if (Input.GetKey(KeyCode.H))
            MoveJoint(5, -jointControlSpeed);

        // Gripper control - Space/Left Shift
        if (gripperJoint != null)
        {
            if (Input.GetKey(KeyCode.Space))
                MoveGripper(-gripperSpeed); // Close
            else if (Input.GetKey(KeyCode.LeftShift))
                MoveGripper(gripperSpeed);  // Open
        }
    }

    void MoveJoint(int jointIndex, float amount)
    {
        if (jointIndex >= robotJoints.Length) return;

        ArticulationBody joint = robotJoints[jointIndex];
        ArticulationDrive drive = joint.xDrive;

        // Get current position in degrees
        float currentPos = joint.jointPosition[0] * Mathf.Rad2Deg;

        // Calculate new position in degrees 
        float newPos = currentPos + amount * Mathf.Rad2Deg;

        // Clamp to joint limits
        newPos = Mathf.Clamp(newPos, -maxJointValue * Mathf.Rad2Deg, maxJointValue * Mathf.Rad2Deg);

        // Apply the new position
        drive.target = newPos;
        drive.stiffness = 10000f;
        drive.damping = 100f;
        drive.forceLimit = 1000f;
        joint.xDrive = drive;

    }

    [SerializeField] private ArticulationBody leftInnerKnuckle;
    [SerializeField] private ArticulationBody rightInnerKnuckle;

    void MoveGripper(float amount)
    {
        float deltaDeg = amount * Mathf.Rad2Deg;

        ApplyGripperRotation(leftInnerKnuckle, deltaDeg);
        ApplyGripperRotation(rightInnerKnuckle, -deltaDeg);
    }

    void ApplyGripperRotation(ArticulationBody joint, float deltaDeg)
    {
        if (joint == null) return;

        var drive = joint.xDrive;
        float currentDeg = joint.jointPosition[0] * Mathf.Rad2Deg;
        float newDeg = Mathf.Clamp(currentDeg + deltaDeg, 0f, 2.5f);

        drive.target = newDeg;
        drive.stiffness = 5000;
        drive.damping = 100;
        drive.forceLimit = 1000;

        joint.xDrive = drive;

        Debug.Log($"[Gripper] {joint.name} → {newDeg:F2}°");
    }



    void ResetSimulation()
    {
        graspingAgent.OnEpisodeBegin();

        Debug.Log("Simulation reset");
    }
}