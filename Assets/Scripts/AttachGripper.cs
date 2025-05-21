using UnityEngine;

public class AttachGripper : MonoBehaviour
{
    public GameObject ur5eRobot;
    public GameObject robotiqGripper;
    public string toolFlangeName = "wrist_3_link"; 

    void Start()
    {
        Debug.Log("AttachGripper script starting...");
        
        if (ur5eRobot == null || robotiqGripper == null)
        {
            Debug.LogError("Assign both UR5e robot and Robotiq gripper in the Inspector!");
            return;
        }
        
        Debug.Log("Full UR5e hierarchy:");
        PrintHierarchy(ur5eRobot.transform, 0);
        Transform toolFlange = FindChildRecursively(ur5eRobot.transform, toolFlangeName);
        
        if (toolFlange != null)
        {
            Debug.Log("Found tool flange: " + toolFlange.name);
            
            robotiqGripper.transform.SetParent(toolFlange, false);
            robotiqGripper.transform.localPosition = Vector3.zero;
            robotiqGripper.transform.localRotation = Quaternion.identity;
            
            Debug.Log("Gripper attached successfully to " + toolFlangeName);
        }
        else
        {
            Debug.LogError("Could not find tool flange named '" + toolFlangeName + "' in the UR5e robot hierarchy!");
        }
    }

    private void PrintHierarchy(Transform transform, int depth)
    {
        string indent = "";
        for (int i = 0; i < depth; i++)
            indent += "  ";
            
        Debug.Log(indent + "- " + transform.name);
        
        foreach (Transform child in transform)
            PrintHierarchy(child, depth + 1);
    }

    private Transform FindChildRecursively(Transform parent, string childName)
    {
        if (parent.name == childName)
            return parent;
        
        foreach (Transform child in parent)
        {
            Transform result = FindChildRecursively(child, childName);
            if (result != null)
                return result;
        }
        
        return null;
    }
}