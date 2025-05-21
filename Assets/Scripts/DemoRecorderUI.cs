using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Demonstrations;


public class DemoRecorderUI : MonoBehaviour
{
    private DemonstrationRecorder recorder;
    private bool isRecording = false;
    private int episodeCount = 0;
    private GraspingAgent agent;

    private void Start()
    {
        recorder = FindObjectOfType<DemonstrationRecorder>();
        agent = FindObjectOfType<GraspingAgent>();

        if (recorder == null)
        {
            Debug.LogError("No DemonstrationRecorder found in the scene!");
        }

        if (agent == null)
        {
            Debug.LogError("No GraspingAgent found in the scene!");
        }
    }

    private void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 10, 300, 200));
        GUILayout.BeginVertical(GUI.skin.box);

        GUILayout.Label("Demonstration Recorder", new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold });

        if (recorder != null)
        {
            string recordingStatus = isRecording ? "Recording Episode " + episodeCount : "Not Recording";
            GUILayout.Label("Status: " + recordingStatus);

            GUILayout.Space(10);

            if (!isRecording)
            {
                if (GUILayout.Button("Start Recording"))
                {
                    StartRecording();
                }
            }
            else
            {
                if (GUILayout.Button("End Episode"))
                {
                    EndEpisode();
                }

                if (GUILayout.Button("Stop Recording"))
                {
                    StopRecording();
                }
            }
        }
        else
        {
            GUILayout.Label("Demonstration Recorder not found!");
        }

        GUILayout.EndVertical();
        GUILayout.EndArea();
    }

    private void StartRecording()
    {
        if (recorder != null)
        {
            var prop = recorder.GetType().GetProperty("recording");
            if (prop != null)
            {
                prop.SetValue(recorder, true);
            }
            else
            {
                Debug.LogError("⚠️ Could not find 'recording' property on DemonstrationRecorder. Check your ML-Agents version.");
            }

            isRecording = true;
            episodeCount = 1;
            Debug.Log("Started recording demonstration.");
        }
    }




    private void EndEpisode()
    {
        if (agent != null)
        {
            agent.EndEpisode();
            episodeCount++;
            Debug.Log("Ended episode and started a new one.");
        }
    }

    private void StopRecording()
    {
        if (recorder != null)
        {
            var prop = recorder.GetType().GetProperty("recording");
            if (prop != null)
            {
                prop.SetValue(recorder, true);
            }
            else
            {
                Debug.LogError("⚠️ Could not find 'recording' property on DemonstrationRecorder. Check your ML-Agents version.");
            }

            isRecording = false;
            episodeCount = 1;
            Debug.Log("Started recording demonstration.");
        }
    }

}