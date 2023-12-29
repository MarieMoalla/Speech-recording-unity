using System.Collections;
using UnityEngine;
using TMPro;

public class ConsoleLogger : MonoBehaviour
{
    public TextMeshProUGUI consoleText; // Reference to the TextMeshPro input field

    void Start()
    {
        consoleText.text="";
        Application.logMessageReceived += HandleLog;
    }

    void OnDestroy()
    {
        Application.logMessageReceived -= HandleLog;
    }

    void HandleLog(string logString, string stackTrace, LogType type)
    {
        string formattedLog = $"{logString}\n";
        
        // Optionally, you can append stack trace or log type information
        // formattedLog += $"StackTrace: {stackTrace}\n";
        // formattedLog += $"LogType: {type}\n";

        UpdateConsoleText(formattedLog);
    }

    void UpdateConsoleText(string log)
    {
        // Update the TextMeshPro input field
        consoleText.text += log;

        // Optionally, you may want to limit the number of lines to keep it readable
        // Adjust 'maxLines' based on your preference
        int maxLines = 20;
        string[] lines = consoleText.text.Split('\n');

        if (lines.Length > maxLines)
        {
            int removeCount = lines.Length - maxLines;
            consoleText.text = string.Join("\n", lines, removeCount, maxLines);
        }
    }
}
