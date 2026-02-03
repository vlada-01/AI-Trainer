// Test page JavaScript

document.addEventListener("DOMContentLoaded", () => {
  const testRunForm = document.getElementById("test-run-form");
  const testRunIdInput = document.getElementById("testRunId");
  const testRunResults = document.getElementById("test-run-results");
  const testRunResultsContent = document.getElementById("testRunResultsContent");

  // Polling interval for test run job
  let testRunPollInterval = null;

  // Store original test error table data for filtering
  let originalTestErrorTable = [];
  let currentTestErrorFilters = {};

  // Helper function to check if job status is terminal
  function isTerminalJobStatus(status) {
    return status === "success" || status === "failed";
  }

  // Helper function to escape HTML
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Helper function to format numbers - round floats to 4 decimals
  function formatNumber(value) {
    if (typeof value === "number" && !isNaN(value) && isFinite(value)) {
      // Check if it's a float (has decimal part)
      if (value % 1 !== 0) {
        // Round to 4 decimals and return as string to preserve formatting
        return value.toFixed(4);
      }
      // It's an integer, return as is (will be converted to string later)
      return value;
    }
    // Not a number, return original value
    return value;
  }

  // Helper function to filter test error table data
  function filterTestErrorTable(data, filters) {
    if (!Array.isArray(data) || data.length === 0) {
      return [];
    }

    return data.filter((dict) => {
      for (const [column, filterValue] of Object.entries(filters)) {
        if (!filterValue || filterValue.trim() === "") {
          continue; // Skip empty filters
        }

        const cellValue = dict[column];
        if (cellValue === undefined || cellValue === null) {
          return false; // If column doesn't exist or is null, exclude row
        }

        // Convert both to strings for case-insensitive comparison
        const cellValueStr = String(cellValue).toLowerCase();
        const filterValueStr = filterValue.trim().toLowerCase();

        // Check if filter value is contained in cell value
        if (!cellValueStr.includes(filterValueStr)) {
          return false;
        }
      }
      return true; // All filters passed
    });
  }

  // Helper function to generate test error table from list of dictionaries
  function generateTestErrorTable(errorTableList, filters = {}) {
    if (!Array.isArray(errorTableList) || errorTableList.length === 0) {
      return {
        html: `<p style="color: #9ca3af; font-style: italic;">No test error data available.</p>`,
        columnHeaders: [],
        rowCount: 0
      };
    }

    // Get column headers from the first dictionary (all dictionaries have the same keys)
    // Use original data to get headers even if filtered data is empty
    const firstDict = errorTableList[0];
    const columnHeaders = Object.keys(firstDict);

    // Apply filters
    const filteredData = filterTestErrorTable(errorTableList, filters);
    const rowCount = filteredData.length;

    // Generate filter row
    let filterRow = "";
    for (const header of columnHeaders) {
      const filterId = `test-error-filter-${escapeHtml(String(header))}`;
      const filterValue = filters[header] || "";
      filterRow += `
        <td style="padding: 0.5rem; background: #111827; border-bottom: 1px solid #374151;">
          <input 
            type="text" 
            id="${filterId}"
            class="test-error-filter-input"
            data-column="${escapeHtml(String(header))}"
            placeholder="Filter ${escapeHtml(String(header))}"
            value="${escapeHtml(filterValue)}"
            style="width: 100%; padding: 0.5rem; background: #1f2937; border: 1px solid #374151; border-radius: 0.25rem; color: #e5e7eb; font-size: 0.875rem; box-sizing: border-box;"
          />
        </td>
      `;
    }

    // Generate header row
    let headerRow = "";
    for (const header of columnHeaders) {
      headerRow += `
        <th style="padding: 0.75rem; text-align: left; color: #e5e7eb; font-weight: 600; border-bottom: 2px solid #374151;">${escapeHtml(String(header))}</th>
      `;
    }

    // Generate data rows
    let dataRows = "";
    if (filteredData.length === 0) {
      // Show a message row spanning all columns when no rows match
      dataRows = `
        <tr>
          <td colspan="${columnHeaders.length}" style="padding: 2rem; text-align: center; color: #9ca3af; font-style: italic; border-bottom: 1px solid #374151;">
            No rows match the current filters.
          </td>
        </tr>
      `;
    } else {
      for (const dict of filteredData) {
        dataRows += "<tr>";
        for (const header of columnHeaders) {
          const value = dict[header];
          // Format value - round floats to 2 decimals, handle objects/arrays
          let displayValue = value;
          if (typeof value === "object" && value !== null) {
            displayValue = JSON.stringify(value, null, 2);
          } else if (typeof value === "number") {
            displayValue = formatNumber(value);
          }
          dataRows += `
            <td style="padding: 0.75rem; border-bottom: 1px solid #374151; color: #d1d5db;">
              ${typeof value === "object" && value !== null 
                ? `<pre style="margin: 0; white-space: pre-wrap; font-size: 0.875rem;">${escapeHtml(String(displayValue))}</pre>`
                : escapeHtml(String(displayValue))
              }
            </td>
          `;
        }
        dataRows += "</tr>";
      }
    }

    return {
      html: `
        <div style="margin-bottom: 0.5rem;">
          <p style="color: #9ca3af; font-size: 0.875rem;">
            Showing ${rowCount} of ${errorTableList.length} rows
            ${Object.keys(filters).some(k => filters[k] && filters[k].trim() !== "") 
              ? '<span style="margin-left: 0.5rem; color: #3b82f6;">(filtered)</span>' 
              : ''
            }
          </p>
        </div>
        <div style="max-height: 500px; overflow-y: auto; border-radius: 0.5rem;" id="test-error-table-container">
          <table style="width: 100%; border-collapse: collapse; background: #1f2937; border-radius: 0.5rem; overflow: hidden;">
            <thead style="position: sticky; top: 0; z-index: 10;">
              <tr style="background: #111827;">
                ${headerRow}
              </tr>
              <tr style="background: #111827;">
                ${filterRow}
              </tr>
            </thead>
            <tbody>
              ${dataRows}
            </tbody>
          </table>
        </div>
      `,
      columnHeaders: columnHeaders,
      rowCount: rowCount
    };
  }

  // Helper function to generate confusion matrix table from List[List[int]]
  function generateConfusionMatrixTable(confusionMatrix) {
    if (!Array.isArray(confusionMatrix) || confusionMatrix.length === 0) {
      return `<p style="color: #9ca3af; font-style: italic;">No confusion matrix data available.</p>`;
    }

    // Validate that all inner lists have the same length
    const numRows = confusionMatrix.length;
    const numCols = confusionMatrix[0]?.length || 0;
    
    if (numCols === 0) {
      return `<p style="color: #9ca3af; font-style: italic;">Invalid confusion matrix data.</p>`;
    }

    // Validate all rows have same length
    for (let i = 0; i < numRows; i++) {
      if (!Array.isArray(confusionMatrix[i]) || confusionMatrix[i].length !== numCols) {
        return `<p style="color: #9ca3af; font-style: italic;">Invalid confusion matrix: rows have different lengths.</p>`;
      }
    }

    // Generate header row with column numbers (0, 1, 2, ...) prefixed with "Predicted"
    let headerRow = '<th style="padding: 0.75rem; text-align: center; color: #e5e7eb; font-weight: 600; border-bottom: 2px solid #374151; background: #111827;"></th>'; // Empty top-left cell
    for (let col = 0; col < numCols; col++) {
      headerRow += `<th style="padding: 0.75rem; text-align: center; color: #e5e7eb; font-weight: 600; border-bottom: 2px solid #374151; background: #111827;">Predicted ${col}</th>`;
    }

    // Generate data rows
    let dataRows = "";
    for (let row = 0; row < numRows; row++) {
      dataRows += "<tr>";
      // Row header (row number) prefixed with "True"
      dataRows += `<td style="padding: 0.75rem; text-align: center; font-weight: 600; color: #e5e7eb; border-right: 2px solid #374151; background: #111827;">True ${row}</td>`;
      // Data cells
      for (let col = 0; col < numCols; col++) {
        const value = confusionMatrix[row][col];
        dataRows += `<td style="padding: 0.75rem; text-align: center; border-bottom: 1px solid #374151; color: #d1d5db;">${escapeHtml(String(value))}</td>`;
      }
      dataRows += "</tr>";
    }

    return `
      <div style="max-height: 500px; overflow-y: auto; border-radius: 0.5rem;">
        <table style="width: 100%; border-collapse: collapse; background: #1f2937; border-radius: 0.5rem; overflow: hidden;">
          <thead style="position: sticky; top: 0; z-index: 10;">
            <tr style="background: #111827;">
              ${headerRow}
            </tr>
          </thead>
          <tbody>
            ${dataRows}
          </tbody>
        </table>
      </div>
    `;
  }

  // Handle test run form submission
  if (testRunForm) {
    testRunForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const runId = testRunIdInput?.value.trim();

      if (!runId) {
        alert("Please enter a Run ID.");
        return;
      }

      // Clear any existing polling
      if (testRunPollInterval) {
        clearInterval(testRunPollInterval);
        testRunPollInterval = null;
      }

      // Reset filters for new test run
      currentTestErrorFilters = {};
      originalTestErrorTable = [];

      // Show results section
      if (testRunResults) {
        testRunResults.style.display = "block";
      }

      // Show loading state
      if (testRunResultsContent) {
        testRunResultsContent.innerHTML = '<p class="muted">Sending test run request...</p>';
      }

      // Send POST request to start test run
      fetch("http://localhost:8000/predict/test", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          run_id: runId,
        }),
      })
        .then(async (response) => {
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
            throw new Error(`HTTP error! status: ${response.status}, detail: ${JSON.stringify(errorData)}`);
          }
          return response.json();
        })
        .then((jobData) => {
          // Display initial response
          displayTestRunResult(jobData);

          // Start polling if job is not terminal
          if (jobData.id && !isTerminalJobStatus(jobData.status)) {
            startTestRunPolling(jobData.id);
          }
        })
        .catch((error) => {
          console.error("Failed to start test run:", error);
          if (testRunResultsContent) {
            testRunResultsContent.innerHTML = `
              <div class="inspect-error">
                <p style="color: #fecaca;">Failed to start test run: ${error.message}</p>
              </div>
            `;
          }
        });
    });
  }

  // Start polling for test run job status
  function startTestRunPolling(jobId) {
    if (testRunPollInterval) {
      clearInterval(testRunPollInterval);
    }

    testRunPollInterval = setInterval(() => {
      fetch(`http://localhost:8000/predict/test_status/${jobId}`, {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
      })
        .then(async (response) => {
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
            throw new Error(`HTTP error! status: ${response.status}, detail: ${JSON.stringify(errorData)}`);
          }
          return response.json();
        })
        .then((jobData) => {
          // Display updated response
          displayTestRunResult(jobData);

          // Stop polling if job is terminal
          if (isTerminalJobStatus(jobData.status)) {
            if (testRunPollInterval) {
              clearInterval(testRunPollInterval);
              testRunPollInterval = null;
            }
          }
        })
        .catch((error) => {
          console.error("Failed to poll test run status:", error);
          if (testRunResultsContent) {
            testRunResultsContent.innerHTML = `
              <div class="inspect-error">
                <p style="color: #fecaca;">Failed to poll test run status: ${error.message}</p>
              </div>
            `;
          }
          // Stop polling on error
          if (testRunPollInterval) {
            clearInterval(testRunPollInterval);
            testRunPollInterval = null;
          }
        });
    }, 2000); // Poll every 2 seconds
  }

  // Display test run result based on status
  function displayTestRunResult(jobData) {
    if (!testRunResultsContent) return;

    const status = jobData.status || "unknown";

    let content = "";

    switch (status) {
      case "pending":
        content = `
          <div style="padding: 1rem; text-align: center;">
            <p style="color: #fbbf24; font-size: 1rem; margin: 0;">⏳ Waiting for loading results of the run...</p>
          </div>
        `;
        break;

      case "in_progress":
        content = `
          <div style="padding: 1rem; text-align: center;">
            <p style="color: #3b82f6; font-size: 1rem; margin: 0;">🔄 Loading results of the run...</p>
          </div>
        `;
        break;

      case "failed":
        const errorDetails = jobData.error || {};
        const errorPretty = JSON.stringify(errorDetails, null, 2);
        content = `
          <div style="padding: 1rem;">
            <p style="color: #ef4444; font-size: 1rem; font-weight: 600; margin-bottom: 1rem;">❌ Error occurred</p>
            <p style="color: #fecaca; margin-bottom: 0.5rem; font-weight: 500;">Details:</p>
            <pre style="background: #1f2937; color: #e5e7eb; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; font-size: 0.875rem; line-height: 1.5; margin: 0;">${errorPretty}</pre>
          </div>
        `;
        break;

      case "success":
        const testMetrics = (jobData.status_details && jobData.status_details.test_metrics) || {};
        const testErrorTable = (jobData.status_details && jobData.status_details.test_error_table) || [];
        const testConfusionMatrix = (jobData.status_details && jobData.status_details.test_confusion_matrix) || null;
        
        // Helper function to generate metrics table
        function generateMetricsTable(metrics, emptyMessage) {
          if (Object.keys(metrics).length === 0) {
            return `<p style="color: #9ca3af; font-style: italic;">${emptyMessage}</p>`;
          }
          
          let tableRows = "";
          for (const [key, value] of Object.entries(metrics)) {
            // Format value - if it's an object/array, stringify it nicely
            let displayValue = value;
            if (typeof value === "object" && value !== null) {
              displayValue = JSON.stringify(value, null, 2);
            }
            tableRows += `
              <tr>
                <td style="padding: 0.75rem; border-bottom: 1px solid #374151; font-weight: 500; color: #e5e7eb;">${escapeHtml(String(key))}</td>
                <td style="padding: 0.75rem; border-bottom: 1px solid #374151; color: #d1d5db;">
                  ${typeof value === "object" && value !== null 
                    ? `<pre style="margin: 0; white-space: pre-wrap; font-size: 0.875rem;">${escapeHtml(String(displayValue))}</pre>`
                    : escapeHtml(String(displayValue))
                  }
                </td>
              </tr>
            `;
          }
          
          return `
            <table style="width: 100%; border-collapse: collapse; background: #1f2937; border-radius: 0.5rem; overflow: hidden;">
              <thead>
                <tr style="background: #111827;">
                  <th style="padding: 0.75rem; text-align: left; color: #e5e7eb; font-weight: 600; border-bottom: 2px solid #374151;">Metric</th>
                  <th style="padding: 0.75rem; text-align: left; color: #e5e7eb; font-weight: 600; border-bottom: 2px solid #374151;">Value</th>
                </tr>
              </thead>
              <tbody>
                ${tableRows}
              </tbody>
            </table>
          `;
        }

        const testMetricsTable = generateMetricsTable(testMetrics, "No test metrics available.");
        
        // Store original data for filtering
        originalTestErrorTable = Array.isArray(testErrorTable) ? [...testErrorTable] : [];
        
        // Generate test error table with current filters (or empty if first time)
        const testErrorTableResult = generateTestErrorTable(originalTestErrorTable, currentTestErrorFilters);
        
        // Generate confusion matrix table if available
        const confusionMatrixTable = testConfusionMatrix 
          ? generateConfusionMatrixTable(testConfusionMatrix)
          : null;

        content = `
          <div style="padding: 1rem;">
            <p style="color: #10b981; font-size: 1rem; font-weight: 600; margin-bottom: 1rem;">✅ Test run completed successfully</p>
            <div style="margin-top: 1rem;">
              <h3 style="color: #e5e7eb; font-size: 1.125rem; margin-bottom: 0.75rem; font-weight: 600;">Test Metrics</h3>
              ${testMetricsTable}
            </div>
            <div style="margin-top: 2rem;">
              <h3 style="color: #e5e7eb; font-size: 1.125rem; margin-bottom: 0.75rem; font-weight: 600;">Test Error Table</h3>
              <div id="test-error-table-wrapper">
                ${testErrorTableResult.html}
              </div>
            </div>
            ${confusionMatrixTable ? `
            <div style="margin-top: 2rem;">
              <h3 style="color: #e5e7eb; font-size: 1.125rem; margin-bottom: 0.75rem; font-weight: 600;">Test Confusion Matrix</h3>
              ${confusionMatrixTable}
            </div>
            ` : ''}
          </div>
        `;
        break;

      default:
        // Fallback: show raw JSON for unknown status
        const prettyJson = JSON.stringify(jobData, null, 2);
        content = `
          <pre style="background: #1f2937; color: #e5e7eb; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; font-size: 0.875rem; line-height: 1.5;">${prettyJson}</pre>
        `;
    }

    testRunResultsContent.innerHTML = content;

    // Set up filter event listeners for test error table if it exists
    if (originalTestErrorTable.length > 0) {
      setupTestErrorTableFilters();
    }
  }

  // Set up filter event listeners for test error table
  function setupTestErrorTableFilters() {
    const filterInputs = document.querySelectorAll(".test-error-filter-input");
    
    // Load existing filter values from currentTestErrorFilters
    filterInputs.forEach((input) => {
      const column = input.getAttribute("data-column");
      if (currentTestErrorFilters[column]) {
        input.value = currentTestErrorFilters[column];
      }
    });

    // Add event listeners for real-time filtering
    filterInputs.forEach((input) => {
      const column = input.getAttribute("data-column");

      // Debounce function to limit filter updates
      let timeoutId = null;
      input.addEventListener("input", (e) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          const filterValue = e.target.value;
          if (filterValue.trim() === "") {
            delete currentTestErrorFilters[column];
          } else {
            currentTestErrorFilters[column] = filterValue;
          }
          updateTestErrorTable();
        }, 300); // 300ms debounce
      });
    });
  }

  // Update test error table with current filters
  function updateTestErrorTable() {
    const wrapper = document.getElementById("test-error-table-wrapper");
    if (!wrapper || originalTestErrorTable.length === 0) return;

    // Regenerate table with current filters
    const testErrorTableResult = generateTestErrorTable(originalTestErrorTable, currentTestErrorFilters);
    wrapper.innerHTML = testErrorTableResult.html;

    // Re-setup filter listeners (since we replaced the HTML)
    setupTestErrorTableFilters();
  }
});
