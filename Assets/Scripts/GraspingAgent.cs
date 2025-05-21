using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;

public class GraspingAgent : Agent
{
    [Header("Robot Configuration")]
    public Transform robotBase;
    public Transform endEffector;
    public ArticulationBody[] robotJoints;
    public ArticulationBody gripperJoint;
    public float maxJointValue = 3.14f; 
    public float maxVelocity = 30f;
    public bool useVelocityControl = false;
    public float gripperClosingSpeed = 0.5f;

    [Header("Environment")]
    public GameObject targetObject;
    public float tableHeight = 0.0f;
    public LayerMask graspableLayer;
    public Transform cameraMount; 
    [Header("Rewards")]
    public float distanceReward = 0.1f;
    public float successReward = 10f;
    public float failurePenalty = -5f;
    public float energyPenalty = 0.001f;
    public float timeoutPenalty = -1f;

    [Header("ROS Integration")]
    public bool enableROS = false;
    public string jointStatesTopic = "/joint_states";
    public string gripCommandTopic = "/grip_command";
    private ROSConnection ros;

    // Internal variables
    private float[] previousJointPositions;
    private bool objectGrasped = false;
    private bool episodeFinished = false;
    private int stepsSinceStart = 0;
    private int maxStepsPerEpisode = 1000;
    private List<GameObject> detectedObjects = new List<GameObject>();
    private Vector3 initialTargetPosition;
    private Vector3 initialEndEffectorPosition;
    private float minDistanceToTarget = float.MaxValue;
    public bool randomizeTarget = true;
    public bool useVisionInput = true;

    // Grasping related variables
    private float gripperOpenValue = 0.04f;
    private Collider[] gripperColliders;
    public ArticulationBody leftFingerJoint;
    public ArticulationBody rightFingerJoint;
    public Collider leftFingerTip;
    public Collider rightFingerTip;
    [SerializeField] private ArticulationBody leftInnerKnuckle;
    [SerializeField] private ArticulationBody rightInnerKnuckle;




    // Visual sensing
    private Camera endEffectorCamera;
    private RenderTexture cameraTexture;
    private int cameraWidth = 84;
    private int cameraHeight = 84;
    // Manual control
    public bool manualControl = false;

    public override void Initialize()
    {
        Debug.Log("robotJoints length: " + robotJoints.Length);
        previousJointPositions = new float[robotJoints.Length];
        for (int i = 0; i < robotJoints.Length; i++)
        {
            previousJointPositions[i] = robotJoints[i].jointPosition[0];
        }

        if (enableROS)
        {
            ros = ROSConnection.GetOrCreateInstance();
            ros.RegisterPublisher<JointStateMsg>(jointStatesTopic);
            ros.Subscribe<BoolMsg>(gripCommandTopic, OnGripCommand);
        }

        if (useVisionInput && cameraMount != null)
        {
            SetupEndEffectorCamera();
        }

        // Find gripper colliders for grasp detection
        if (gripperJoint != null)
        {
            gripperColliders = gripperJoint.GetComponentsInChildren<Collider>();
        }

        foreach (var joint in robotJoints)
        {
            var drive = joint.xDrive;
            drive.stiffness = 10000f;
            drive.damping = 100f;
            drive.forceLimit = 1000f;
            joint.xDrive = drive;
        }

        if (gripperJoint != null)
        {
            var gripDrive = gripperJoint.xDrive;
            gripDrive.stiffness = 5000f;
            gripDrive.damping = 100f;
            gripDrive.forceLimit = 1000f;
            gripperJoint.xDrive = gripDrive;
        }

        if (leftInnerKnuckle != null)
        {
            var drive = leftInnerKnuckle.xDrive;
            drive.lowerLimit = 0f;
            drive.upperLimit = 2.5f;
            drive.stiffness = 5000f;
            drive.damping = 100f;
            drive.forceLimit = 1000f;
            leftInnerKnuckle.xDrive = drive;
        }

        if (rightInnerKnuckle != null)
        {
            var drive = rightInnerKnuckle.xDrive;
            drive.lowerLimit = -2.5f;
            drive.upperLimit = 0f;
            drive.stiffness = 5000f;
            drive.damping = 100f;
            drive.forceLimit = 1000f;
            rightInnerKnuckle.xDrive = drive;
        }
    }

    private void SetupEndEffectorCamera()
    {
        // Create camera if it doesn't exist yet
        if (endEffectorCamera == null)
        {
            GameObject camObj = new GameObject("EndEffectorCamera");
            camObj.transform.SetParent(cameraMount, false);
            camObj.transform.localPosition = new Vector3(0, 0, 0);
            camObj.transform.localRotation = Quaternion.Euler(90, 0, 0);

            endEffectorCamera = camObj.AddComponent<Camera>();
            endEffectorCamera.nearClipPlane = 0.01f;
            endEffectorCamera.farClipPlane = 1.0f;

            // Create render texture
            cameraTexture = new RenderTexture(cameraWidth, cameraHeight, 24);
            endEffectorCamera.targetTexture = cameraTexture;
        }
    }

    public override void OnEpisodeBegin()
    {
        if (manualControl) return;
        objectGrasped = false;
        episodeFinished = false;
        stepsSinceStart = 0;
        minDistanceToTarget = float.MaxValue;

        ResetRobotPose();

        if (randomizeTarget && targetObject != null)
        {
            RandomizeTargetPosition();
        }

        if (targetObject != null)
        {
            initialTargetPosition = targetObject.transform.position;
        }

        if (endEffector != null)
        {
            initialEndEffectorPosition = endEffector.position;
        }
    }

    private void ResetRobotPose()
    {
        if (manualControl) return;
        // Reset each joint to a default position
        float[] defaultPositions = { 0f, -1.5f, 1.5f, -1.57f, -1.57f, 0f };

        for (int i = 0; i < robotJoints.Length; i++)
        {
            ArticulationDrive drive = robotJoints[i].xDrive;
            float targetPos = (i < defaultPositions.Length) ? defaultPositions[i] : 0f;
            drive.target = targetPos * Mathf.Rad2Deg;
            robotJoints[i].xDrive = drive;
            previousJointPositions[i] = targetPos;
        }

        // Open gripper
        if (gripperJoint != null)
        {
            ArticulationDrive drive = gripperJoint.xDrive;
            drive.target = gripperOpenValue * Mathf.Rad2Deg;
            gripperJoint.xDrive = drive;
        }
    }

    private void RandomizeTargetPosition()
    {
        if (targetObject == null) return;

        // Get Rigidbody if exists
        Rigidbody rb = targetObject.GetComponent<Rigidbody>();

        // Calculate random position on table
        float tableRadius = 0.3f;
        Vector3 randomPos = robotBase.position + new Vector3(
            UnityEngine.Random.Range(-tableRadius, tableRadius),
            tableHeight + 0.05f,
            UnityEngine.Random.Range(-tableRadius, tableRadius)
        );

        targetObject.transform.position = randomPos;


        targetObject.transform.rotation = UnityEngine.Random.rotation;


        if (rb != null)
        {
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Add joint positions and velocities
        foreach (ArticulationBody joint in robotJoints)
        {
            sensor.AddObservation(joint.jointPosition[0] / maxJointValue);
            sensor.AddObservation(joint.jointVelocity[0] / maxVelocity);
        }

        // Add gripper state
        if (gripperJoint != null)
        {
            sensor.AddObservation(gripperJoint.jointPosition[0] / gripperOpenValue);
        }

        // Target object observations
        if (targetObject != null)
        {
            // Get object position relative to robot base
            Vector3 relativePos = robotBase.InverseTransformPoint(targetObject.transform.position);
            sensor.AddObservation(relativePos / 1.0f);

            // Get end effector position relative to robot base
            Vector3 endEffectorRelativePos = robotBase.InverseTransformPoint(endEffector.position);
            sensor.AddObservation(endEffectorRelativePos / 1.0f);

            // Observe vector from end effector to target
            Vector3 endEffectorToTarget = targetObject.transform.position - endEffector.position;
            sensor.AddObservation(endEffectorToTarget / 1.0f);

            // Distance to target
            float distanceToTarget = Vector3.Distance(endEffector.position, targetObject.transform.position);
            sensor.AddObservation(distanceToTarget / 1.0f);
        }
        else
        {
            // If no target, use zeroes for target-related observations
            sensor.AddObservation(new Vector3(0, 0, 0)); // Relative position
            sensor.AddObservation(new Vector3(0, 0, 0)); // End effector position
            sensor.AddObservation(new Vector3(0, 0, 0)); // Vector to target
            sensor.AddObservation(0); // Distance
        }

        sensor.AddObservation(objectGrasped ? 1.0f : 0.0f);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (manualControl) return;
        // Extract continuous actions 
        float[] actions = actionBuffers.ContinuousActions.Array;

        // Control robot joints
        int actionIndex = 0;
        for (int i = 0; i < robotJoints.Length; i++)
        {
            ArticulationBody joint = robotJoints[i];


            if (useVelocityControl)
            {
                // Velocity control
                float targetVelocity = actions[actionIndex] * maxVelocity;
                ArticulationDrive drive = joint.xDrive;
                drive.targetVelocity = targetVelocity;
                joint.xDrive = drive;
            }
            else
            {
                // Position control 
                float targetPosition = previousJointPositions[i] + actions[actionIndex] * 0.1f;
                targetPosition = Mathf.Clamp(targetPosition, -maxJointValue, maxJointValue);
                Debug.Log($"[Joint {i}] Target pos: {targetPosition}, Raw input: {actions[actionIndex]}");


                ArticulationDrive drive = joint.xDrive;
                drive.target = targetPosition * Mathf.Rad2Deg;
                joint.xDrive = drive;

                // Store for next step
                previousJointPositions[i] = targetPosition;
            }

            actionIndex++;
        }

        // Control gripper
        if (gripperJoint != null && actionIndex < actions.Length)
        {
            float gripperAction = actions[actionIndex];

            // Map [-1, 1] to [0, 0.04] (closed to open)
            float targetGripperPosition = Mathf.Lerp(0.0f, 0.04f, (gripperAction + 1f) / 2f);

            ArticulationDrive drive = gripperJoint.xDrive;
            drive.target = targetGripperPosition * Mathf.Rad2Deg;
            gripperJoint.xDrive = drive;
        }

        // Calculate reward
        if (targetObject != null)
        {
            // Distance-based reward (negative reward for being far from target)
            float distance = Vector3.Distance(endEffector.position, targetObject.transform.position);
            float distanceRewardValue = Mathf.Exp(-distance * 5) * distanceReward;
            AddReward(distanceRewardValue);

            // Update minimum distance for progress tracking
            if (distance < minDistanceToTarget)
            {
                minDistanceToTarget = distance;
                AddReward(0.01f); // Small reward for getting closer than ever before
            }

            // Check if object is grasped
            CheckGrasping();

            // Height reward 
            if (objectGrasped)
            {
                float heightAboveTable = targetObject.transform.position.y - tableHeight;
                float heightReward = Mathf.Clamp01(heightAboveTable / 0.1f) * 0.005f;
                AddReward(heightReward);

                // Success condition: object lifted above certain height
                if (heightAboveTable > 0.2f)
                {
                    AddReward(successReward);
                    episodeFinished = true;
                    EndEpisode();
                }
            }
        }

        // Energy penalty to discourage excessive movement
        float energyUsed = 0f;
        foreach (ArticulationBody joint in robotJoints)
        {
            energyUsed += Mathf.Abs(joint.jointVelocity[0]);
        }
        AddReward(-energyUsed * energyPenalty);

        // If using ROS bridge, publish joint states
        if (enableROS)
        {
            PublishJointStates();
        }

        // Track steps and end episode if it's taking too long
        stepsSinceStart++;
        if (stepsSinceStart >= maxStepsPerEpisode && !episodeFinished)
        {
            AddReward(timeoutPenalty);
            EndEpisode();
        }
    }

    private void CheckGrasping()
    {
        if (targetObject == null || leftFingerJoint == null || rightFingerJoint == null) return;

        // 1. Check if both fingers are closed
        float leftClosed = leftFingerJoint.jointPosition[0];
        float rightClosed = rightFingerJoint.jointPosition[0];
        bool isClosed = leftClosed < 0.5f && rightClosed < 0.5f;


        // 2. Check if both fingertips are touching the object
        bool leftTouch = false;
        bool rightTouch = false;

        Collider[] leftHits = Physics.OverlapBox(
            leftFingerTip.bounds.center,
            leftFingerTip.bounds.extents * 0.9f,
            leftFingerTip.transform.rotation
        );

        foreach (var hit in leftHits)
        {
            if (hit.gameObject == targetObject)
            {
                leftTouch = true;
                break;
            }
        }

        Collider[] rightHits = Physics.OverlapBox(
            rightFingerTip.bounds.center,
            rightFingerTip.bounds.extents * 0.9f,
            rightFingerTip.transform.rotation
        );

        foreach (var hit in rightHits)
        {
            if (hit.gameObject == targetObject)
            {
                rightTouch = true;
                break;
            }
        }

        Debug.Log($"[CheckGrasp] left: {leftClosed}, right: {rightClosed}, touching? {leftTouch} / {rightTouch}");

        // 3. Successful grasp = closed AND both fingers touch
        if (isClosed && leftTouch && rightTouch)
        {
            if (!objectGrasped)
            {
                Debug.DrawLine(leftFingerTip.bounds.center, rightFingerTip.bounds.center, Color.red, 1f);
                Debug.Log("Grasping detected!");
                objectGrasped = true;

                targetObject.transform.SetParent(gripperJoint.transform);

                var rb = targetObject.GetComponent<Rigidbody>();
                if (rb) rb.isKinematic = true;

                Debug.Log("✅ Object grasped (Robotiq)");
            }
        }
        else
        {
            // If we had grasped, but released — drop it
            if (objectGrasped)
            {
                objectGrasped = false;
                targetObject.transform.SetParent(null);

                var rb = targetObject.GetComponent<Rigidbody>();
                if (rb) rb.isKinematic = false;

                Debug.Log("🛑 Object released");
            }
        }

        if (!targetObject.TryGetComponent<FixedJoint>(out var joint))
        {
            var fixedJoint = targetObject.AddComponent<FixedJoint>();
            fixedJoint.connectedBody = gripperJoint.GetComponent<Rigidbody>();
        }

        Debug.Log($"LeftTip hit count: {leftHits.Length}, RightTip hit count: {rightHits.Length}");

    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        Debug.Log("Heuristic called");
        if (Input.GetKey(KeyCode.Q)) Debug.Log("Q pressed");

        var continuousActions = actionsOut.ContinuousActions;

        // Initialize all actions to 0 (no movement)
        for (int i = 0; i < robotJoints.Length + 1; i++) // +1 for gripper
        {
            continuousActions[i] = 0f;
        }

        // Control each joint with keyboard keys
        // Joint 0 (shoulder) - Q/A keys
        if (Input.GetKey(KeyCode.Q))
            continuousActions[0] = 1f;
        else if (Input.GetKey(KeyCode.A))
            continuousActions[0] = -1f;

        // Joint 1 (upper arm) - W/S keys
        if (Input.GetKey(KeyCode.W))
            continuousActions[1] = 1f;
        else if (Input.GetKey(KeyCode.S))
            continuousActions[1] = -1f;

        // Joint 2 (forearm) - E/D keys
        if (Input.GetKey(KeyCode.E))
            continuousActions[2] = 1f;
        else if (Input.GetKey(KeyCode.D))
            continuousActions[2] = -1f;

        // Joint 3 (wrist_1) - R/F keys
        if (Input.GetKey(KeyCode.R))
            continuousActions[3] = 1f;
        else if (Input.GetKey(KeyCode.F))
            continuousActions[3] = -1f;

        // Joint 4 (wrist_2) - T/G keys
        if (Input.GetKey(KeyCode.T))
            continuousActions[4] = 1f;
        else if (Input.GetKey(KeyCode.G))
            continuousActions[4] = -1f;

        // Joint 5 (wrist_3) - Y/H keys
        if (Input.GetKey(KeyCode.Y))
            continuousActions[5] = 1f;
        else if (Input.GetKey(KeyCode.H))
            continuousActions[5] = -1f;

        // Last action (gripper control) - Spacebar to close, Left Shift to open
        if (Input.GetKey(KeyCode.Space))
            continuousActions[6] = 1f;  // Close gripper
        else if (Input.GetKey(KeyCode.LeftShift))
            continuousActions[6] = -1f; // Open gripper

        if (Input.anyKey)
        {
            string actionValues = "Actions: ";
            for (int i = 0; i < continuousActions.Length; i++)
            {
                actionValues += continuousActions[i].ToString("F1") + ", ";
            }
            Debug.Log(actionValues);
        }
    }

    private void PublishJointStates()
    {
        if (ros == null) return;

        JointStateMsg jointStateMsg = new JointStateMsg();

        jointStateMsg.name = new string[robotJoints.Length + (gripperJoint != null ? 1 : 0)];
        jointStateMsg.position = new double[robotJoints.Length + (gripperJoint != null ? 1 : 0)];
        jointStateMsg.velocity = new double[robotJoints.Length + (gripperJoint != null ? 1 : 0)];
        jointStateMsg.effort = new double[robotJoints.Length + (gripperJoint != null ? 1 : 0)];

        for (int i = 0; i < robotJoints.Length; i++)
        {
            jointStateMsg.name[i] = "joint_" + i;
            jointStateMsg.position[i] = robotJoints[i].jointPosition[0];
            jointStateMsg.velocity[i] = robotJoints[i].jointVelocity[0];
            jointStateMsg.effort[i] = 0;
        }

        if (gripperJoint != null)
        {
            int index = robotJoints.Length;
            jointStateMsg.name[index] = "gripper_joint";
            jointStateMsg.position[index] = gripperJoint.jointPosition[0];
            jointStateMsg.velocity[index] = gripperJoint.jointVelocity[0];
            jointStateMsg.effort[index] = 0;
        }

        ros.Publish(jointStatesTopic, jointStateMsg);
    }

    private void OnGripCommand(BoolMsg msg)
    {
        if (leftInnerKnuckle == null || rightInnerKnuckle == null)
        {
            Debug.LogWarning("❌ Inner knuckles not assigned!");
            return;
        }

        float targetDeg = msg.data ? 2.5f : 0f;

        SetFingerTarget(leftInnerKnuckle, targetDeg);
        SetFingerTarget(rightInnerKnuckle, -targetDeg);

        Debug.Log($"🦾 Gripper {(msg.data ? "OPEN" : "CLOSE")} | TargetDeg: {targetDeg}");
    }


    private void SetFingerTarget(ArticulationBody joint, float angleDeg)
    {
        var drive = joint.xDrive;
        drive.target = angleDeg;
        drive.stiffness = 5000;
        drive.damping = 100;
        drive.forceLimit = 1000;
        joint.xDrive = drive;
    }

    // Unity event functions for handling collisions
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject == targetObject)
        {
            // Check which robot joint collided
            var contact = collision.GetContact(0);
            var source = contact.thisCollider;

            foreach (var joint in robotJoints)
            {
                if (source.transform.IsChildOf(joint.transform))
                {
                    Debug.Log($"🟥 COLLISION: {joint.name} hit target cube!");
                    AddReward(0.1f);
                    return;
                }
            }

            if (gripperJoint != null && source.transform.IsChildOf(gripperJoint.transform))
            {
                Debug.Log($"🟥 COLLISION: Gripper hit target cube!");
                AddReward(0.1f);
                return;
            }

            Debug.Log("🟥 COLLISION: Unknown part hit target cube!");
            AddReward(0.05f);
        }
        else if (collision.gameObject.CompareTag("Table"))
        {
            if (collision.relativeVelocity.magnitude > 1.0f)
            {
                AddReward(-0.2f);
            }
        }
    }

    void Update()
    {

        // Vision based detection of objects 
        if (useVisionInput && endEffectorCamera != null)
        {
            // process the camera image
            // For now, it's a placeholder for vision-based detection

            // TODO:
            // 1. Capture the camera image
            // 2. Run object detection on it (possibly with ML)
            // 3. Update the detectedObjects list
        }
    }

}