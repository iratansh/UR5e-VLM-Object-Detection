using UnityEngine;

public class DirectAttachGripper : MonoBehaviour
{
    public Transform endEffectorLink; 
    public GameObject robotiqGripper;
    
    void Start()
    {
        Debug.Log("Starting gripper attachment...");
        
        if (endEffectorLink == null)
        {
            Debug.LogError("End effector link not assigned! Please assign wrist_3_link in the Inspector.");
            return;
        }
        
        if (robotiqGripper == null)
        {
            Debug.LogError("Robotiq gripper not assigned! Please assign the gripper in the Inspector.");
            return;
        }
        
        Debug.Log("Attaching gripper to: " + endEffectorLink.name);
        robotiqGripper.transform.SetParent(endEffectorLink, false);
        
        robotiqGripper.transform.localPosition = Vector3.zero;
        robotiqGripper.transform.localRotation = Quaternion.identity;
        
        Debug.Log("Gripper attached successfully. Check hierarchy to confirm.");
    }
    
    public void AttachGripperNow()
    {
        if (endEffectorLink != null && robotiqGripper != null)
        {
            Debug.Log("Manually attaching gripper...");
            robotiqGripper.transform.SetParent(endEffectorLink, false);
            robotiqGripper.transform.localPosition = Vector3.zero;
            robotiqGripper.transform.localRotation = Quaternion.identity;
            Debug.Log("Manual attachment complete.");
        }
        else
        {
            Debug.LogError("Cannot attach: Missing references to end effector or gripper.");
        }
    }
}