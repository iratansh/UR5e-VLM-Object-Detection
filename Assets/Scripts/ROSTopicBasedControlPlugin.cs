using System;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.UrdfImporter;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;

public class ROSTopicBasedControlPlugin : MonoBehaviour
{

    ROSConnection ros;

    public ArticulationBody[] Joints;

    public string jointStatesTopic = "/joint_states";
    public string jointCommandsTopic = "/joint_command";

    public float frequency = 20f;

    private float TimeElapsed;
    private JointStateMsg stateMsg;
    private JointStateMsg commandMsg;

    private bool recvdCommandJointOrder = false;
    private int[] commandJointOrder;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<JointStateMsg>(gameObject.transform.root.name + jointStatesTopic);
        ros.Subscribe<JointStateMsg>(gameObject.transform.root.name + jointCommandsTopic, CommandCallback);

        stateMsg = new JointStateMsg();
        commandMsg = new JointStateMsg();

        stateMsg.name = new string[Joints.Length];

        for (uint i = 0; i < Joints.Length; i++)
        {
            stateMsg.name[i] = Joints[i].GetComponent<UrdfJoint>().jointName;
        }

        stateMsg.position = new double[Joints.Length];
        stateMsg.velocity = new double[Joints.Length];
        stateMsg.effort = new double[Joints.Length];

        stateMsg.header = new HeaderMsg();
#if ROS2
#else
        stateMsg.header.seq = 0;
#endif

    }

    private void Update()
    {
        TimeElapsed += Time.deltaTime;

        if (TimeElapsed > 1 / frequency)
        {
            float time = Time.time;
#if ROS2
            stateMsg.header.stamp.sec = (int)Math.Truncate(time);
#else
            stateMsg.header.stamp.sec = (uint)Math.Truncate(time);
            stateMsg.header.seq++;
#endif
            stateMsg.header.stamp.nanosec = (uint)((time - stateMsg.header.stamp.sec) * 1e+9);

            for (uint i = 0; i < Joints.Length; i++)
            {
                stateMsg.position[i] = (double)Joints[i].jointPosition[0];
                stateMsg.velocity[i] = (double)Joints[i].jointVelocity[0];
                stateMsg.effort[i] = (double)Joints[i].jointForce[0];
            }

            ros.Publish(gameObject.transform.root.name + jointStatesTopic, stateMsg);
            TimeElapsed = 0;
        }
    }

    void CommandCallback(JointStateMsg msg)
    {
        if (msg.position.Length != Joints.Length)
        {
            Debug.LogWarning("⚠️ Received message contains invalid number of joints.");
            return;
        }

        if (!recvdCommandJointOrder)
        {
            commandJointOrder = new int[msg.name.Length];

            for (int i = 0; i < msg.name.Length; i++)
            {
                string remappedName = msg.name[i]
                    .Replace("shoulder_pan_joint", "shoulder_link")
                    .Replace("shoulder_lift_joint", "upper_arm_link")
                    .Replace("elbow_joint", "forearm_link")
                    .Replace("wrist_1_joint", "wrist_1_link")
                    .Replace("wrist_2_joint", "wrist_2_link")
                    .Replace("wrist_3_joint", "wrist_3_link");

                bool found = false;
                for (int jx = 0; jx < Joints.Length; jx++)
                {
                    if (Joints[jx].name == remappedName)
                    {
                        commandJointOrder[i] = jx;
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    Debug.LogWarning($"⚠️ Could not map ROS joint '{msg.name[i]}' to Unity joint '{remappedName}'");
                }
            }

            recvdCommandJointOrder = true;
        }

        for (int i = 0; i < msg.position.Length; i++)
        {
            int jointIndex = commandJointOrder[i];
            float targetAngle = (float)msg.position[i] * Mathf.Rad2Deg;

            var drive = Joints[jointIndex].xDrive;
            drive.target = targetAngle;
            drive.stiffness = 10000f;
            drive.damping = 500f;
            drive.forceLimit = 10000f;
            Joints[jointIndex].xDrive = drive;

            Debug.Log($"✅ Moving joint {msg.name[i]} to {targetAngle:F1}°");
        }
    }


}