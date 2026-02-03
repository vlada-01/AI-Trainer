// Inspect page JavaScript

document.addEventListener("DOMContentLoaded", () => {
  const inspectRunForm = document.getElementById("inspect-run-form");
  const runIdInput = document.getElementById("runId");
  const experimentsList = document.getElementById("experimentsList");
  const inspectRunResults = document.getElementById("inspect-run-results");
  const inspectRunResultsContent = document.getElementById("inspectRunResultsContent");

  // Polling interval for inspect run job
  let inspectRunPollInterval = null;

  // Store original validation error table data for filtering
  let originalValErrorTable = [];
  let currentValErrorFilters = {};

  // Post-processing: methods and drag state
  let postProcessingMethods = [];
  let postProcessingDragIndex = null;

  const postProcessingDisplayNames = {
    global_threshold: "Global Threshold",
    calibration: "Calibration",
  };

  // Fine-tuning: transformations
  let fineTuningTransforms = [];
  let fineTuningDragIndex = null;

  // Fine-tuning: layers
  let fineTuningLayers = [];
  let fineTuningLayerDragIndex = null;
  let fineTuningUseTorchLayers = false;
  // Track layer metadata: type (backbone/new), freeze status, and original_id
  let ftLayerDetails = []; // Array of {type: "backbone" | "new", freeze: boolean, original_id: number | null}

  // Fine-tuning: training configuration
  let fineTuningTrainCfg = {
    epochs: 10,
    num_of_iters: 100,
    optimizer: { type: "", args: {} },
    lr_decay: null,
    loss_fn: { type: "", args: {} },
  };

  // Helper function to get default layer configuration
  function getDefaultFineTuningLayer(type = "Linear") {
    switch (type) {
      case "Conv2d":
        return {
          type: "Conv2d",
          in_channels: 3,
          out_channels: 32,
          kernel_size: 3,
          padding: 1,
          stride: 1,
        };
      case "MaxPool2d":
        return {
          type: "MaxPool2d",
          kernel_size: 2,
          stride: 2,
        };
      case "ReLU":
        return {
          type: "ReLU",
        };
      case "BatchNorm2d":
        return {
          type: "BatchNorm2d",
          num_features: 32,
        };
      case "Flatten":
        return {
          type: "Flatten",
        };
      case "Dropout":
        return {
          type: "Dropout",
          p: 0.5,
        };
      case "Linear":
      default:
        return {
          type: "Linear",
          in_features: 128,
          out_features: 64,
          bias: true,
        };
    }
  }

  // Function to render fine-tuning layers
  function renderFineTuningLayers() {
    const container = document.getElementById("fineTuningLayersContainer");
    if (!container) return;
    container.innerHTML = "";

    if (fineTuningLayers.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No layers added yet. Click \"Add layer\" to start.";
      container.appendChild(empty);
      return;
    }

    fineTuningLayers.forEach((layer, index) => {
      const card = document.createElement("div");
      card.className = "layer-card";
      card.draggable = true;
      card.dataset.index = String(index);
      
      // Get layer details (backbone/new and freeze status)
      const layerDetail = ftLayerDetails[index] || { type: "new", freeze: false, original_id: null };
      const isBackbone = layerDetail.type === "backbone";
      const isFrozen = layerDetail.freeze;
      
      // Add background color and border based on layer type for visual distinction
      if (isBackbone) {
        card.style.backgroundColor = "#1f2937"; // Standard dark background for backbone
        card.style.borderLeft = "3px solid #6b7280"; // Gray border for backbone
      } else {
        card.style.backgroundColor = "#111827"; // Slightly different background for new layers
        card.style.borderLeft = "3px solid #3b82f6"; // Blue border for new layers
      }

      let paramsHtml = "";

      if (layer.type === "Conv2d") {
        paramsHtml = `
          <div class="field-group">
            <label>in_channels</label>
            <input type="number" min="1" data-field="in_channels" data-index="${index}" value="${
              layer.in_channels ?? ""
            }" />
          </div>
          <div class="field-group">
            <label>out_channels</label>
            <input type="number" min="1" data-field="out_channels" data-index="${index}" value="${
              layer.out_channels ?? ""
            }" />
          </div>
          <div class="field-group">
            <label>kernel_size</label>
            <input type="number" min="1" data-field="kernel_size" data-index="${index}" value="${
              layer.kernel_size ?? ""
            }" />
          </div>
          <div class="field-group">
            <label>padding</label>
            <input type="number" min="0" data-field="padding" data-index="${index}" value="${
              layer.padding ?? 0
            }" />
          </div>
          <div class="field-group">
            <label>stride</label>
            <input type="number" min="1" data-field="stride" data-index="${index}" value="${
              layer.stride ?? 1
            }" />
          </div>
        `;
      } else if (layer.type === "MaxPool2d") {
        paramsHtml = `
          <div class="field-group">
            <label>kernel_size</label>
            <input type="number" min="1" data-field="kernel_size" data-index="${index}" value="${
              layer.kernel_size ?? ""
            }" />
          </div>
          <div class="field-group">
            <label>stride (optional)</label>
            <input type="number" min="1" data-field="stride" data-index="${index}" value="${
              layer.stride ?? ""
            }" />
          </div>
        `;
      } else if (layer.type === "Linear") {
        paramsHtml = `
          <div class="field-group">
            <label>in_features</label>
            <input type="number" min="1" data-field="in_features" data-index="${index}" value="${
              layer.in_features ?? ""
            }" />
          </div>
          <div class="field-group">
            <label>out_features</label>
            <input type="number" min="1" data-field="out_features" data-index="${index}" value="${
              layer.out_features ?? ""
            }" />
          </div>
          <div class="field-group inline">
            <div>
              <label>bias</label>
              <p class="field-help">Include bias term</p>
            </div>
            <label class="switch">
              <input type="checkbox" data-field="bias" data-index="${index}" ${
                layer.bias ? "checked" : ""
              } />
              <span class="slider"></span>
            </label>
          </div>
        `;
      } else if (layer.type === "Dropout") {
        paramsHtml = `
          <div class="field-group">
            <label>p (drop probability)</label>
            <input type="number" min="0" max="1" step="0.05" data-field="p" data-index="${index}" value="${
              layer.p ?? 0.5
            }" />
          </div>
        `;
      } else if (layer.type === "ReLU") {
        paramsHtml = `
          <p class="muted">ReLU has no additional parameters.</p>
        `;
      } else if (layer.type === "BatchNorm2d") {
        paramsHtml = `
          <div class="field-group">
            <label>num_features</label>
            <input type="number" min="1" data-field="num_features" data-index="${index}" value="${
              layer.num_features ?? ""
            }" />
          </div>
        `;
      } else if (layer.type === "Flatten") {
        paramsHtml = `
          <p class="muted">Flatten has no additional parameters.</p>
        `;
      }

      card.innerHTML = `
        <div class="layer-header">
          <div class="layer-header-main">
            <div class="layer-header-top-row" style="display: flex; align-items: center; gap: 0.5rem;">
              <span class="layer-index">Layer ${index + 1}</span>
              <button 
                type="button" 
                class="btn small ${isFrozen ? 'secondary' : 'primary'}" 
                data-toggle-freeze="${index}"
                title="${isFrozen ? 'Frozen (click to unfreeze)' : 'Unfrozen (click to freeze)'}"
                style="padding: 0.25rem 0.5rem; font-size: 0.75rem;"
              >
                ${isFrozen ? '🔒 Frozen' : '🔓 Unfrozen'}
              </button>
              ${isBackbone ? '<span style="font-size: 0.75rem; color: #9ca3af;">(Backbone)</span>' : '<span style="font-size: 0.75rem; color: #3b82f6;">(New)</span>'}
            </div>
            <div class="layer-type-row">
              <label class="layer-type-label" for="fine-tuning-layer-type-${index}">Layer type</label>
              <select id="fine-tuning-layer-type-${index}" data-field="type" data-index="${index}">
              <option value="Conv2d" ${layer.type === "Conv2d" ? "selected" : ""}>Conv2d</option>
              <option value="MaxPool2d" ${
                layer.type === "MaxPool2d" ? "selected" : ""
              }>MaxPool2d</option>
              <option value="Linear" ${layer.type === "Linear" ? "selected" : ""}>Linear</option>
              <option value="ReLU" ${layer.type === "ReLU" ? "selected" : ""}>ReLU</option>
              <option value="BatchNorm2d" ${layer.type === "BatchNorm2d" ? "selected" : ""}>BatchNorm2d</option>
              <option value="Flatten" ${layer.type === "Flatten" ? "selected" : ""}>Flatten</option>
              <option value="Dropout" ${layer.type === "Dropout" ? "selected" : ""}>Dropout</option>
            </select>
            </div>
          </div>
          <div class="layer-actions">
            <span class="drag-handle" title="Drag to reorder">⋮⋮</span>
            <button type="button" data-remove="${index}">Remove</button>
          </div>
        </div>
        <div class="layer-fields">
          ${paramsHtml}
        </div>
      `;

      container.appendChild(card);
    });
  }

  const transformDisplayNames = {
    to_tensor: "Convert to Tensor",
    img_to_tensor: "Image to Tensor",
    normalize_mean_std: "Normalize (Mean & Std)",
    normalize_min_max: "Normalize (Min-Max)",
    normalize_l1_l2: "L1/L2 Normalization",
    random_horizontal_flip: "Random Horizontal Flip",
    random_vertical_flip: "Random Vertical Flip",
    random_rotation: "Random Rotation",
  };

  // Available transformations (sync with Prepare Dataset section)
  // To add a new transformation, add it here and it will appear in Fine Tuning
  const availableTransforms = [
    { value: "to_tensor", label: "Convert to Tensor" },
    { value: "img_to_tensor", label: "Image to Tensor" },
    { value: "normalize_mean_std", label: "Normalize (Mean & Std)" },
    { value: "normalize_min_max", label: "Normalize (Min-Max)" },
    { value: "normalize_l1_l2", label: "L1/L2 Normalization" },
    { value: "random_horizontal_flip", label: "Random Horizontal Flip" },
    { value: "random_vertical_flip", label: "Random Vertical Flip" },
    { value: "random_rotation", label: "Random Rotation" },
  ];

  // Function to populate Fine Tuning transform select dropdown dynamically
  function populateFineTuningTransformSelect() {
    const select = document.getElementById("fineTuningTransformSelect");
    if (!select) return;

    // Clear existing options except the first one
    select.innerHTML = '<option value="">Add transformation...</option>';

    // Add all available transforms
    availableTransforms.forEach((transform) => {
      const option = document.createElement("option");
      option.value = transform.value;
      option.textContent = transform.label;
      select.appendChild(option);
    });
  }

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

  // --- Post-processing ---
  function renderPostProcessingList() {
    const list = document.getElementById("postProcessingList");
    if (!list) return;
    list.innerHTML = "";

    if (postProcessingMethods.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No post-processing methods added yet. Select a method to add it to the list.";
      list.appendChild(empty);
      return;
    }

    postProcessingMethods.forEach((m, index) => {
      const item = document.createElement("div");
      item.className = "transform-item post-processing-item";
      item.draggable = true;
      item.dataset.index = String(index);
      const label = postProcessingDisplayNames[m.type] || m.type;
      const accuracyValue = m.type === "global_threshold" && m.accuracy !== undefined ? m.accuracy : 0.0;
      const accuracyInputHtml = m.type === "global_threshold"
        ? `
        <label style="margin-left: 10px; display: inline-flex; align-items: center; gap: 5px;">
          accuracy:
          <input
            type="number"
            step="0.01"
            class="post-processor-accuracy-input"
            data-index="${index}"
            value="${accuracyValue}"
            style="width: 80px; padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid #374151; background: #1f2937; color: #e5e7eb; font-size: 0.875rem;"
          />
        </label>
        `
        : "";
      item.innerHTML = `
        <span class="drag-handle" title="Drag to reorder">⋮⋮</span>
        <span class="transform-name">${escapeHtml(label)}</span>
        ${accuracyInputHtml}
        <button type="button" class="btn small secondary" data-remove="${index}">Remove</button>
      `;
      list.appendChild(item);
    });
  }

  function addPostProcessingMethod(type) {
    if (!type || (type !== "global_threshold" && type !== "calibration")) return;
    if (type === "global_threshold") {
      postProcessingMethods.push({ type, accuracy: 0.0 });
    } else {
      postProcessingMethods.push({ type });
    }
    renderPostProcessingList();
  }

  function removePostProcessingMethod(index) {
    if (index < 0 || index >= postProcessingMethods.length) return;
    postProcessingMethods.splice(index, 1);
    renderPostProcessingList();
  }

  function setupPostProcessing() {
    const select = document.getElementById("postProcessingMethodSelect");
    const list = document.getElementById("postProcessingList");
    const sendBtn = document.getElementById("sendPostProcessorBtn");

    if (select) {
      select.addEventListener("change", () => {
        const v = select.value;
        if (v) {
          addPostProcessingMethod(v);
          select.value = "";
        }
      });
    }

    if (sendBtn) {
      sendBtn.addEventListener("click", () => {
        // Get the new run name
        const newRunNameInput = document.getElementById("newRunName");
        const newRunName = newRunNameInput ? newRunNameInput.value.trim() : "";

        // Build JSON request body for backend
        // Map type values: global_threshold -> GlobalThreshold, calibration -> Calibration
        const typeMapping = {
          global_threshold: "GlobalThreshold",
          calibration: "Calibration",
        };
        
        const payload = {
          new_run_name: newRunName || undefined,
          post_processors: postProcessingMethods.map((m) => {
            const base = { type: typeMapping[m.type] || m.type };
            if (m.type === "global_threshold" && m.accuracy !== undefined) {
              base.accuracy = Number(m.accuracy);
            }
            return base;
          }),
        };

        // Clear any existing polling
        if (inspectRunPollInterval) {
          clearInterval(inspectRunPollInterval);
          inspectRunPollInterval = null;
        }

        // Reset filters for new results
        currentValErrorFilters = {};
        originalValErrorTable = [];

        // Show results section
        if (inspectRunResults) {
          inspectRunResults.style.display = "block";
        }

        // Show loading state
        if (inspectRunResultsContent) {
          inspectRunResultsContent.innerHTML = '<p class="muted">Sending post-process run request...</p>';
        }

        // Send POST request to start post-process run
        fetch("http://localhost:8000/train/post_process_run", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify(payload),
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
            displayInspectRunResult(jobData);

            // Start polling if job is not terminal
            if (jobData.id && !isTerminalJobStatus(jobData.status)) {
              startPostProcessRunPolling(jobData.id);
            }
          })
          .catch((error) => {
            console.error("Failed to start post-process run:", error);
            if (inspectRunResultsContent) {
              inspectRunResultsContent.innerHTML = `
                <div class="inspect-error">
                  <p style="color: #fecaca;">Failed to start post-process run: ${error.message}</p>
                </div>
              `;
            }
          });
      });
    }

    if (!list) return;

    list.addEventListener("click", (e) => {
      if (e.target.dataset.remove !== undefined) {
        const item = e.target.closest(".post-processing-item");
        if (!item) return;
        const index = parseInt(item.dataset.index, 10);
        removePostProcessingMethod(index);
      }
    });

    list.addEventListener("input", (e) => {
      if (e.target.classList.contains("post-processor-accuracy-input")) {
        const index = parseInt(e.target.dataset.index, 10);
        if (index >= 0 && index < postProcessingMethods.length && postProcessingMethods[index].type === "global_threshold") {
          const val = parseFloat(e.target.value);
          postProcessingMethods[index].accuracy = isNaN(val) ? 0.0 : val;
        }
      }
    });

    list.addEventListener("change", (e) => {
      if (e.target.classList.contains("post-processor-accuracy-input")) {
        const index = parseInt(e.target.dataset.index, 10);
        if (index >= 0 && index < postProcessingMethods.length && postProcessingMethods[index].type === "global_threshold") {
          const val = parseFloat(e.target.value);
          postProcessingMethods[index].accuracy = isNaN(val) ? 0.0 : val;
        }
      }
    });

    list.addEventListener("dragstart", (e) => {
      const item = e.target.closest(".post-processing-item");
      if (!item) return;
      postProcessingDragIndex = parseInt(item.dataset.index, 10);
      item.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
    });

    list.addEventListener("dragend", (e) => {
      const item = e.target.closest(".post-processing-item");
      if (item) item.classList.remove("dragging");
      postProcessingDragIndex = null;
      Array.from(list.children).forEach((el) => el.classList.remove("drag-over"));
    });

    list.addEventListener("dragover", (e) => {
      e.preventDefault();
      const item = e.target.closest(".post-processing-item");
      if (!item || postProcessingDragIndex === null) return;
      Array.from(list.children).forEach((el) => el.classList.remove("drag-over"));
      item.classList.add("drag-over");
    });

    list.addEventListener("drop", (e) => {
      e.preventDefault();
      const target = e.target.closest(".post-processing-item");
      if (!target || postProcessingDragIndex === null) return;
      const targetIndex = parseInt(target.dataset.index, 10);
      if (Number.isNaN(targetIndex) || targetIndex === postProcessingDragIndex) return;
      const [moved] = postProcessingMethods.splice(postProcessingDragIndex, 1);
      postProcessingMethods.splice(targetIndex, 0, moved);
      postProcessingDragIndex = null;
      renderPostProcessingList();
    });
  }

  // --- Fine-tuning ---
  function renderFineTuningTransformList() {
    const list = document.getElementById("fineTuningTransformList");
    if (!list) return;
    list.innerHTML = "";

    if (fineTuningTransforms.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No transformations added yet. Select a transformation to add it to the list.";
      list.appendChild(empty);
      return;
    }

    fineTuningTransforms.forEach((transformObj, index) => {
      const item = document.createElement("div");
      item.className = "transform-item";
      item.draggable = true;
      item.dataset.index = String(index);
      const transformName = transformObj.type;
      const displayName = transformDisplayNames[transformName] || transformName.replace(/_/g, " ");

      let innerHTML = `
        <span class="drag-handle" title="Drag to reorder">⋮⋮</span>
        <span class="transform-name">${escapeHtml(displayName)}</span>
      `;

      // Add value inputs for transforms that need them
      if (transformName === "normalize_l1_l2") {
        const pValue = transformObj.value && transformObj.value.p !== undefined ? transformObj.value.p : 1;
        innerHTML += `
          <label style="margin-left: 10px; display: inline-flex; align-items: center; gap: 5px;">
            p:
            <select class="fine-tuning-value-input" data-index="${index}" style="padding: 2px 5px; border-radius: 4px; border: 1px solid #374151; background: #1f2937; color: #e5e7eb;">
              <option value="1" ${pValue === 1 ? 'selected' : ''}>1</option>
              <option value="2" ${pValue === 2 ? 'selected' : ''}>2</option>
            </select>
          </label>
        `;
      } else if (transformName === "random_horizontal_flip") {
        const pValue = transformObj.value && transformObj.value.p !== undefined ? transformObj.value.p : 0.5;
        innerHTML += `
          <label style="margin-left: 10px; display: inline-flex; align-items: center; gap: 5px;">
            p:
            <input type="number" step="0.01" min="0" max="1" class="fine-tuning-value-input" data-index="${index}" value="${pValue}" style="width: 60px; padding: 2px 5px; border-radius: 4px; border: 1px solid #374151; background: #1f2937; color: #e5e7eb;" />
          </label>
        `;
      } else if (transformName === "random_vertical_flip") {
        const pValue = transformObj.value && transformObj.value.p !== undefined ? transformObj.value.p : 0.5;
        innerHTML += `
          <label style="margin-left: 10px; display: inline-flex; align-items: center; gap: 5px;">
            p:
            <input type="number" step="0.01" min="0" max="1" class="fine-tuning-value-input" data-index="${index}" value="${pValue}" style="width: 60px; padding: 2px 5px; border-radius: 4px; border: 1px solid #374151; background: #1f2937; color: #e5e7eb;" />
          </label>
        `;
      } else if (transformName === "random_rotation") {
        const alphaValue = transformObj.value && transformObj.value.alpha !== undefined ? transformObj.value.alpha : 15;
        innerHTML += `
          <label style="margin-left: 10px; display: inline-flex; align-items: center; gap: 5px;">
            alpha:
            <input type="number" step="1" min="0" class="fine-tuning-value-input" data-index="${index}" value="${alphaValue}" style="width: 60px; padding: 2px 5px; border-radius: 4px; border: 1px solid #374151; background: #1f2937; color: #e5e7eb;" />
          </label>
        `;
      }

      innerHTML += `<button type="button" class="btn small secondary" data-remove="${index}">Remove</button>`;
      item.innerHTML = innerHTML;
      list.appendChild(item);
    });
  }

  function addFineTuningTransform(transformName) {
    if (!transformName) return;

    const exists = fineTuningTransforms.some(t => t.type === transformName);
    if (exists) {
      alert(`"${transformName}" is already in the list.`);
      return;
    }

    let transformObj;
    if (transformName === "normalize_l1_l2") {
      transformObj = { type: "normalize_l1_l2", value: { p: 1 } };
    } else if (transformName === "img_to_tensor") {
      transformObj = { type: "img_to_tensor", value: true };
    } else if (transformName === "random_horizontal_flip") {
      transformObj = { type: "random_horizontal_flip", value: { p: 0.5 } };
    } else if (transformName === "random_vertical_flip") {
      transformObj = { type: "random_vertical_flip", value: { p: 0.5 } };
    } else if (transformName === "random_rotation") {
      transformObj = { type: "random_rotation", value: { alpha: 15 } };
    } else {
      transformObj = { type: transformName, value: true };
    }

    fineTuningTransforms.push(transformObj);
    renderFineTuningTransformList();
  }

  function removeFineTuningTransform(index) {
    if (index < 0 || index >= fineTuningTransforms.length) return;
    fineTuningTransforms.splice(index, 1);
    renderFineTuningTransformList();
  }

  function setupFineTuning() {
    const select = document.getElementById("fineTuningTransformSelect");
    const list = document.getElementById("fineTuningTransformList");
    const sendBtn = document.getElementById("sendFineTuningBtn");

    if (select) {
      select.addEventListener("change", () => {
        const v = select.value;
        if (v) {
          addFineTuningTransform(v);
          select.value = "";
        }
      });
    }

    if (sendBtn) {
      sendBtn.addEventListener("click", () => {
        // Get the new run name
        const newRunNameInput = document.getElementById("fineTuningRunName");
        const newRunName = newRunNameInput ? newRunNameInput.value.trim() : "";

        // Build new_train_cfg object
        const newTrainCfg = {
          epochs: fineTuningTrainCfg.epochs,
          num_of_iters: fineTuningTrainCfg.num_of_iters,
          optimizer: fineTuningTrainCfg.optimizer.type ? fineTuningTrainCfg.optimizer : undefined,
          lr_decay: fineTuningTrainCfg.lr_decay,
          loss_fn: fineTuningTrainCfg.loss_fn.type ? fineTuningTrainCfg.loss_fn : undefined,
        };

        // Remove undefined/null fields
        if (!newTrainCfg.optimizer) delete newTrainCfg.optimizer;
        if (!newTrainCfg.lr_decay || (newTrainCfg.lr_decay && !newTrainCfg.lr_decay.type)) {
          delete newTrainCfg.lr_decay;
        }
        if (!newTrainCfg.loss_fn) delete newTrainCfg.loss_fn;

        // Build new_layers_cfg object
        const newLayersCfg = {
          use_torch_layers: fineTuningUseTorchLayers,
          layers: fineTuningLayers.map((layer) => ({ ...layer })),
          ft_layers_details: ftLayerDetails.map((detail) => ({ ...detail })),
        };

        // Debug: Log ft_layers_details separately to verify original_id
        console.log("ft_layers_details:", JSON.stringify(newLayersCfg.ft_layers_details, null, 2));

        // Build JSON request body
        const payload = {
          new_run_name: newRunName || undefined,
          new_ds_cfg: {
            new_train_transform: fineTuningTransforms.map((t) => ({
              name: t.type,
              value: t.value
            })),
          },
          new_layers_cfg: newLayersCfg,
          new_train_cfg: Object.keys(newTrainCfg).length > 0 ? newTrainCfg : undefined,
        };

        // Print fine-tune configuration to console before sending
        console.log("Fine-tune configuration (sending to endpoint):", JSON.stringify(payload, null, 2));

        // Clear any existing polling
        if (inspectRunPollInterval) {
          clearInterval(inspectRunPollInterval);
          inspectRunPollInterval = null;
        }

        // Reset filters for new results
        currentValErrorFilters = {};
        originalValErrorTable = [];

        // Show results section
        if (inspectRunResults) {
          inspectRunResults.style.display = "block";
        }

        // Show loading state
        if (inspectRunResultsContent) {
          inspectRunResultsContent.innerHTML = '<p class="muted">Sending fine-tuning run request...</p>';
        }

        // Send POST request to start fine-tuning run
        fetch("http://localhost:8000/train/fine_tune_run", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify(payload),
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
            displayFineTuneRunResult(jobData);

            // Start polling if job is not terminal
            if (jobData.id && !isTerminalJobStatus(jobData.status)) {
              startFineTuneRunPolling(jobData.id);
            }
          })
          .catch((error) => {
            console.error("Failed to start fine-tuning run:", error);
            if (inspectRunResultsContent) {
              inspectRunResultsContent.innerHTML = `
                <div class="inspect-error">
                  <p style="color: #fecaca;">Failed to start fine-tuning run: ${error.message}</p>
                </div>
              `;
            }
          });
      });
    }

    if (!list) return;

    list.addEventListener("click", (e) => {
      if (e.target.dataset.remove !== undefined) {
        const item = e.target.closest(".transform-item");
        if (!item) return;
        const index = parseInt(item.dataset.index, 10);
        removeFineTuningTransform(index);
      }
    });

    // Handle value input changes
    list.addEventListener("change", (e) => {
      if (e.target.classList.contains("fine-tuning-value-input")) {
        const index = parseInt(e.target.dataset.index, 10);
        if (index >= 0 && index < fineTuningTransforms.length) {
          const transform = fineTuningTransforms[index];
          if (transform.type === "normalize_l1_l2") {
            transform.value = { p: parseInt(e.target.value, 10) };
          } else if (transform.type === "random_horizontal_flip" || transform.type === "random_vertical_flip") {
            transform.value = { p: parseFloat(e.target.value) };
          } else if (transform.type === "random_rotation") {
            transform.value = { alpha: parseInt(e.target.value, 10) };
          }
        }
      }
    });

    list.addEventListener("input", (e) => {
      if (e.target.classList.contains("fine-tuning-value-input") && e.target.type === "number") {
        const index = parseInt(e.target.dataset.index, 10);
        if (index >= 0 && index < fineTuningTransforms.length) {
          const transform = fineTuningTransforms[index];
          if (transform.type === "random_horizontal_flip" || transform.type === "random_vertical_flip") {
            transform.value = { p: parseFloat(e.target.value) };
          } else if (transform.type === "random_rotation") {
            transform.value = { alpha: parseInt(e.target.value, 10) };
          }
        }
      }
    });

    list.addEventListener("dragstart", (e) => {
      const item = e.target.closest(".transform-item");
      if (!item) return;
      fineTuningDragIndex = parseInt(item.dataset.index, 10);
      item.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
    });

    list.addEventListener("dragend", (e) => {
      const item = e.target.closest(".transform-item");
      if (item) item.classList.remove("dragging");
      fineTuningDragIndex = null;
      Array.from(list.children).forEach((el) => el.classList.remove("drag-over"));
    });

    list.addEventListener("dragover", (e) => {
      e.preventDefault();
      const item = e.target.closest(".transform-item");
      if (!item || fineTuningDragIndex === null) return;
      Array.from(list.children).forEach((el) => el.classList.remove("drag-over"));
      item.classList.add("drag-over");
    });

    list.addEventListener("drop", (e) => {
      e.preventDefault();
      const target = e.target.closest(".transform-item");
      if (!target || fineTuningDragIndex === null) return;
      const targetIndex = parseInt(target.dataset.index, 10);
      if (Number.isNaN(targetIndex) || targetIndex === fineTuningDragIndex) return;
      const [moved] = fineTuningTransforms.splice(fineTuningDragIndex, 1);
      fineTuningTransforms.splice(targetIndex, 0, moved);
      fineTuningDragIndex = null;
      renderFineTuningTransformList();
    });

    // Setup training parameters
    const epochsInput = document.getElementById("fineTuningEpochs");
    const numOfItersInput = document.getElementById("fineTuningNumOfIters");
    const optimizerType = document.getElementById("fineTuningOptimizerType");
    const optimizerArgs = document.getElementById("fineTuningOptimizerArgs");
    const optimizerArgsError = document.getElementById("fineTuningOptimizerArgsError");
    const lrDecayType = document.getElementById("fineTuningLrDecayType");
    const lrDecayArgs = document.getElementById("fineTuningLrDecayArgs");
    const lrDecayArgsError = document.getElementById("fineTuningLrDecayArgsError");
    const lossFnType = document.getElementById("fineTuningLossFnType");
    const lossFnArgs = document.getElementById("fineTuningLossFnArgs");
    const lossFnArgsError = document.getElementById("fineTuningLossFnArgsError");

    // Helper function to validate JSON input
    function validateJsonInput(textarea, errorElement) {
      const value = textarea.value.trim();
      if (value === "") {
        errorElement.style.display = "none";
        return true;
      }
      try {
        JSON.parse(value);
        errorElement.style.display = "none";
        return true;
      } catch (e) {
        errorElement.textContent = `Invalid JSON: ${e.message}`;
        errorElement.style.display = "block";
        return false;
      }
    }

    // Helper function to update config from JSON
    function updateConfigFromJson(textarea, configPath, errorElement) {
      const value = textarea.value.trim();
      if (value === "") {
        const keys = configPath.split('.');
        let obj = fineTuningTrainCfg;
        for (let i = 0; i < keys.length - 1; i++) {
          obj = obj[keys[i]];
        }
        obj[keys[keys.length - 1]] = {};
        return;
      }
      
      if (validateJsonInput(textarea, errorElement)) {
        try {
          const parsed = JSON.parse(value);
          const keys = configPath.split('.');
          let obj = fineTuningTrainCfg;
          for (let i = 0; i < keys.length - 1; i++) {
            obj = obj[keys[i]];
          }
          obj[keys[keys.length - 1]] = parsed;
        } catch (e) {
          // Already handled by validateJsonInput
        }
      }
    }

    // Epochs
    if (epochsInput) {
      epochsInput.addEventListener("input", () => {
        const value = parseInt(epochsInput.value, 10);
        if (!isNaN(value) && value >= 1) {
          fineTuningTrainCfg.epochs = value;
        } else {
          fineTuningTrainCfg.epochs = 10;
        }
      });
    }

    // Number of iterations
    if (numOfItersInput) {
      numOfItersInput.addEventListener("input", () => {
        const value = parseInt(numOfItersInput.value, 10);
        if (!isNaN(value) && value >= 1) {
          fineTuningTrainCfg.num_of_iters = value;
        } else {
          fineTuningTrainCfg.num_of_iters = 100;
        }
      });
    }

    // Optimizer type
    if (optimizerType) {
      optimizerType.addEventListener("input", () => {
        fineTuningTrainCfg.optimizer.type = optimizerType.value;
      });
    }

    // Optimizer args
    if (optimizerArgs) {
      optimizerArgs.addEventListener("input", () => {
        updateConfigFromJson(optimizerArgs, "optimizer.args", optimizerArgsError);
      });
      optimizerArgs.addEventListener("blur", () => {
        validateJsonInput(optimizerArgs, optimizerArgsError);
      });
    }

    // Learning rate decay type
    if (lrDecayType) {
      lrDecayType.addEventListener("input", () => {
        if (lrDecayType.value.trim() === "") {
          fineTuningTrainCfg.lr_decay = null;
        } else {
          if (!fineTuningTrainCfg.lr_decay) {
            fineTuningTrainCfg.lr_decay = { type: "", args: {} };
          }
          fineTuningTrainCfg.lr_decay.type = lrDecayType.value;
        }
      });
    }

    // Learning rate decay args
    if (lrDecayArgs) {
      lrDecayArgs.addEventListener("input", () => {
        if (lrDecayType && lrDecayType.value.trim() !== "") {
          if (!fineTuningTrainCfg.lr_decay) {
            fineTuningTrainCfg.lr_decay = { type: lrDecayType.value, args: {} };
          }
          updateConfigFromJson(lrDecayArgs, "lr_decay.args", lrDecayArgsError);
        }
      });
      lrDecayArgs.addEventListener("blur", () => {
        validateJsonInput(lrDecayArgs, lrDecayArgsError);
      });
    }

    // Loss function type
    if (lossFnType) {
      lossFnType.addEventListener("input", () => {
        fineTuningTrainCfg.loss_fn.type = lossFnType.value;
      });
    }

    // Loss function args
    if (lossFnArgs) {
      lossFnArgs.addEventListener("input", () => {
        updateConfigFromJson(lossFnArgs, "loss_fn.args", lossFnArgsError);
      });
      lossFnArgs.addEventListener("blur", () => {
        validateJsonInput(lossFnArgs, lossFnArgsError);
      });
    }

    // Setup collapsible sections (optimizer, lrDecay, lossFn)
    document.querySelectorAll(".transform-toggle[data-config]").forEach((toggle) => {
      if (toggle.dataset.hasListener === "true") return;
      toggle.dataset.hasListener = "true";
      toggle.addEventListener("click", () => {
        const configType = toggle.dataset.config;
        let content;
        if (configType === "fineTuningOptimizer") {
          content = document.getElementById("fineTuningOptimizerContent");
        } else if (configType === "fineTuningLrDecay") {
          content = document.getElementById("fineTuningLrDecayContent");
        } else if (configType === "fineTuningLossFn") {
          content = document.getElementById("fineTuningLossFnContent");
        }
        const icon = toggle.querySelector(".toggle-icon");
        
        if (content) {
          const isHidden = content.hidden;
          content.hidden = !isHidden;
          icon.textContent = isHidden ? "▼" : "▶";
        }
      });
    });

    // Setup layers
    const addLayerBtn = document.getElementById("fineTuningAddLayerBtn");
    const layersContainer = document.getElementById("fineTuningLayersContainer");
    const usePresetLayers = document.getElementById("fineTuningUsePresetLayers");

    if (addLayerBtn) {
      addLayerBtn.addEventListener("click", () => {
        fineTuningLayers.push(getDefaultFineTuningLayer("Linear"));
        // Add new layer detail: type "new", freeze false, original_id null (not a default layer)
        ftLayerDetails.push({ type: "new", freeze: false, original_id: null });
        renderFineTuningLayers();
      });
    }

    if (usePresetLayers) {
      usePresetLayers.addEventListener("change", () => {
        fineTuningUseTorchLayers = usePresetLayers.checked;
      });
    }

    if (layersContainer) {
      // Drag & drop reordering
      layersContainer.addEventListener("dragstart", (event) => {
        const card = event.target.closest(".layer-card");
        if (!card) return;
        fineTuningLayerDragIndex = parseInt(card.dataset.index, 10);
        card.classList.add("dragging");
        event.dataTransfer.effectAllowed = "move";
      });

      layersContainer.addEventListener("dragend", (event) => {
        const card = event.target.closest(".layer-card");
        if (card) {
          card.classList.remove("dragging");
        }
        fineTuningLayerDragIndex = null;
        Array.from(layersContainer.children).forEach((child) =>
          child.classList.remove("drag-over")
        );
      });

      layersContainer.addEventListener("dragover", (event) => {
        event.preventDefault();
        const card = event.target.closest(".layer-card");
        if (!card || fineTuningLayerDragIndex === null) return;

        Array.from(layersContainer.children).forEach((child) =>
          child.classList.remove("drag-over")
        );
        card.classList.add("drag-over");
      });

      layersContainer.addEventListener("drop", (event) => {
        event.preventDefault();
        const targetCard = event.target.closest(".layer-card");
        if (!targetCard || fineTuningLayerDragIndex === null) return;

        const targetIndex = parseInt(targetCard.dataset.index, 10);
        if (Number.isNaN(targetIndex) || targetIndex === fineTuningLayerDragIndex) return;

        // Move layer
        const [movedLayer] = fineTuningLayers.splice(fineTuningLayerDragIndex, 1);
        fineTuningLayers.splice(targetIndex, 0, movedLayer);
        
        // Move corresponding layer detail
        const [movedDetail] = ftLayerDetails.splice(fineTuningLayerDragIndex, 1);
        ftLayerDetails.splice(targetIndex, 0, movedDetail);

        fineTuningLayerDragIndex = null;
        renderFineTuningLayers();
      });

      layersContainer.addEventListener("click", (event) => {
        const target = event.target;

        if (target.dataset.remove !== undefined) {
          const index = parseInt(target.dataset.remove, 10);
          fineTuningLayers.splice(index, 1);
          ftLayerDetails.splice(index, 1); // Remove corresponding detail
          renderFineTuningLayers();
          return;
        }

        // Handle freeze toggle
        if (target.dataset.toggleFreeze !== undefined) {
          const index = parseInt(target.dataset.toggleFreeze, 10);
          if (index >= 0 && index < ftLayerDetails.length) {
            ftLayerDetails[index].freeze = !ftLayerDetails[index].freeze;
            renderFineTuningLayers();
          }
          return;
        }
      });

      layersContainer.addEventListener("change", (event) => {
        const target = event.target;
        const index = parseInt(target.dataset.index, 10);
        const field = target.dataset.field;

        if (Number.isNaN(index) || !field) return;

        if (field === "type") {
          // When changing type, reset to sensible defaults for that type
          fineTuningLayers[index] = getDefaultFineTuningLayer(target.value);
          // Preserve the layer detail (type and freeze status) when changing layer type
          // The detail is already at the correct index, no need to change it
          renderFineTuningLayers();
          return;
        }

        if (target.type === "checkbox") {
          fineTuningLayers[index][field] = target.checked;
        } else if (target.type === "number") {
          const raw = target.value;
          // Allow optional fields (like stride for MaxPool2d) to be empty
          fineTuningLayers[index][field] = raw === "" ? null : Number(raw);
        } else {
          fineTuningLayers[index][field] = target.value;
        }
      });
    }
  }

  // Helper function to filter validation error table data
  function filterValErrorTable(data, filters) {
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

  // Helper function to generate validation error table from list of dictionaries
  function generateValErrorTable(errorTableList, filters = {}) {
    if (!Array.isArray(errorTableList) || errorTableList.length === 0) {
      return {
        html: `<p style="color: #9ca3af; font-style: italic;">No validation error data available.</p>`,
        columnHeaders: [],
        rowCount: 0
      };
    }

    // Get column headers from the first dictionary (all dictionaries have the same keys)
    // Use original data to get headers even if filtered data is empty
    const firstDict = errorTableList[0];
    const columnHeaders = Object.keys(firstDict);

    // Apply filters
    const filteredData = filterValErrorTable(errorTableList, filters);
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

  // Handle run inspection form submission
  if (inspectRunForm) {
    inspectRunForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const runId = runIdInput?.value.trim();

      if (!runId) {
        alert("Please enter a Run ID.");
        return;
      }

      // Clear any existing polling
      if (inspectRunPollInterval) {
        clearInterval(inspectRunPollInterval);
        inspectRunPollInterval = null;
      }

      // Reset filters for new inspect run
      currentValErrorFilters = {};
      originalValErrorTable = [];

      // Show results section
      if (inspectRunResults) {
        inspectRunResults.style.display = "block";
      }

      // Show loading state
      if (inspectRunResultsContent) {
        inspectRunResultsContent.innerHTML = '<p class="muted">Sending inspect run request...</p>';
      }

      // Send POST request to start inspect run
      fetch("http://localhost:8000/train/inspect_run", {
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
          displayInspectRunResult(jobData);

          // Start polling if job is not terminal
          if (jobData.id && !isTerminalJobStatus(jobData.status)) {
            startInspectRunPolling(jobData.id);
          }
        })
        .catch((error) => {
          console.error("Failed to start inspect run:", error);
          if (inspectRunResultsContent) {
            inspectRunResultsContent.innerHTML = `
              <div class="inspect-error">
                <p style="color: #fecaca;">Failed to start inspect run: ${error.message}</p>
              </div>
            `;
          }
        });
    });
  }

  // Start polling for inspect run job status
  function startInspectRunPolling(jobId) {
    if (inspectRunPollInterval) {
      clearInterval(inspectRunPollInterval);
    }

    inspectRunPollInterval = setInterval(() => {
      fetch(`http://localhost:8000/train/inspect_run/${jobId}`, {
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
          displayInspectRunResult(jobData);

          // Stop polling if job is terminal
          if (isTerminalJobStatus(jobData.status)) {
            if (inspectRunPollInterval) {
              clearInterval(inspectRunPollInterval);
              inspectRunPollInterval = null;
            }
          }
        })
        .catch((error) => {
          console.error("Failed to poll inspect run status:", error);
          if (inspectRunResultsContent) {
            inspectRunResultsContent.innerHTML = `
              <div class="inspect-error">
                <p style="color: #fecaca;">Failed to poll inspect run status: ${error.message}</p>
              </div>
            `;
          }
          // Stop polling on error
          if (inspectRunPollInterval) {
            clearInterval(inspectRunPollInterval);
            inspectRunPollInterval = null;
          }
        });
    }, 2000); // Poll every 2 seconds
  }

  // Display fine-tune run result based on status
  function displayFineTuneRunResult(jobData) {
    if (!inspectRunResultsContent) return;

    const status = jobData.status || "unknown";

    let content = "";

    switch (status) {
      case "pending":
        content = `
          <div style="padding: 1rem; text-align: center;">
            <p style="color: #fbbf24; font-size: 1rem; margin: 0;">⏳ Waiting for fine-tuning to start...</p>
          </div>
        `;
        break;

      case "in_progress":
        content = `
          <div style="padding: 1rem; text-align: center;">
            <p style="color: #3b82f6; font-size: 1rem; margin: 0;">🔄 Fine-tuning in progress...</p>
          </div>
        `;
        break;

      case "failed":
        const errorDetails = jobData.error || {};
        const errorPretty = JSON.stringify(errorDetails, null, 2);
        content = `
          <div style="padding: 1rem;">
            <p style="color: #ef4444; font-size: 1rem; font-weight: 600; margin-bottom: 1rem;">❌ Fine-tuning failed</p>
            <p style="color: #fecaca; margin-bottom: 0.5rem; font-weight: 500;">Details:</p>
            <pre style="background: #1f2937; color: #e5e7eb; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; font-size: 0.875rem; line-height: 1.5; margin: 0;">${errorPretty}</pre>
          </div>
        `;
        break;

      case "success":
        content = `
          <div style="padding: 1rem; text-align: center;">
            <p style="color: #10b981; font-size: 1rem; font-weight: 600; margin: 0;">✅ Fine Tune is successfully done</p>
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

    inspectRunResultsContent.innerHTML = content;
  }

  // Start polling for fine-tune run job status
  function startFineTuneRunPolling(jobId) {
    if (inspectRunPollInterval) {
      clearInterval(inspectRunPollInterval);
    }

    inspectRunPollInterval = setInterval(() => {
      fetch(`http://localhost:8000/train/fine_tune_status/${jobId}`, {
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
          // Display updated response (overwrites Inspect Run Results section)
          displayFineTuneRunResult(jobData);

          // Stop polling if job is terminal
          if (isTerminalJobStatus(jobData.status)) {
            if (inspectRunPollInterval) {
              clearInterval(inspectRunPollInterval);
              inspectRunPollInterval = null;
            }
          }
        })
        .catch((error) => {
          console.error("Failed to poll fine-tune run status:", error);
          if (inspectRunResultsContent) {
            inspectRunResultsContent.innerHTML = `
              <div class="inspect-error">
                <p style="color: #fecaca;">Failed to poll fine-tune run status: ${error.message}</p>
              </div>
            `;
          }
          // Stop polling on error
          if (inspectRunPollInterval) {
            clearInterval(inspectRunPollInterval);
            inspectRunPollInterval = null;
          }
        });
    }, 2000); // Poll every 2 seconds
  }

  // Start polling for post-process run job status
  function startPostProcessRunPolling(jobId) {
    if (inspectRunPollInterval) {
      clearInterval(inspectRunPollInterval);
    }

    inspectRunPollInterval = setInterval(() => {
      fetch(`http://localhost:8000/train/post_process_run/${jobId}`, {
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
          // Display updated response (overwrites Inspect Run Results section)
          displayInspectRunResult(jobData);

          // Stop polling if job is terminal
          if (isTerminalJobStatus(jobData.status)) {
            if (inspectRunPollInterval) {
              clearInterval(inspectRunPollInterval);
              inspectRunPollInterval = null;
            }
          }
        })
        .catch((error) => {
          console.error("Failed to poll post-process run status:", error);
          if (inspectRunResultsContent) {
            inspectRunResultsContent.innerHTML = `
              <div class="inspect-error">
                <p style="color: #fecaca;">Failed to poll post-process run status: ${error.message}</p>
              </div>
            `;
          }
          // Stop polling on error
          if (inspectRunPollInterval) {
            clearInterval(inspectRunPollInterval);
            inspectRunPollInterval = null;
          }
        });
    }, 2000); // Poll every 2 seconds
  }

  // Display inspect run result based on status
  function displayInspectRunResult(jobData) {
    if (!inspectRunResultsContent) return;

    const status = jobData.status || "unknown";
    let layerCfg = null; // Will be set in success case

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
        const valMetrics = (jobData.status_details && jobData.status_details.val_metrics) || {};
        const valErrorTable = (jobData.status_details && jobData.status_details.val_error_table) || [];
        const valConfusionMatrix = (jobData.status_details && jobData.status_details.val_confusion_matrix) || null;
        layerCfg = (jobData.status_details && jobData.status_details.layer_cfg) || null;
        
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

        const valMetricsTable = generateMetricsTable(valMetrics, "No validation metrics available.");
        
        // Store original data for filtering
        originalValErrorTable = Array.isArray(valErrorTable) ? [...valErrorTable] : [];
        
        // Generate validation error table with current filters (or empty if first time)
        const valErrorTableResult = generateValErrorTable(originalValErrorTable, currentValErrorFilters);
        
        // Generate confusion matrix table if available
        const confusionMatrixTable = valConfusionMatrix 
          ? generateConfusionMatrixTable(valConfusionMatrix)
          : null;

        content = `
          <div style="padding: 1rem;">
            <p style="color: #10b981; font-size: 1rem; font-weight: 600; margin-bottom: 1rem;">✅ Run inspection completed successfully</p>
            <div style="margin-top: 1rem;">
              <h3 style="color: #e5e7eb; font-size: 1.125rem; margin-bottom: 0.75rem; font-weight: 600;">Validation Metrics</h3>
              ${valMetricsTable}
            </div>
            <div style="margin-top: 2rem;">
              <h3 style="color: #e5e7eb; font-size: 1.125rem; margin-bottom: 0.75rem; font-weight: 600;">Validation Error Table</h3>
              <div id="test-error-table-wrapper">
                ${valErrorTableResult.html}
              </div>
            </div>
            ${confusionMatrixTable ? `
            <div style="margin-top: 2rem;">
              <h3 style="color: #e5e7eb; font-size: 1.125rem; margin-bottom: 0.75rem; font-weight: 600;">Validation Confusion Matrix</h3>
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

    inspectRunResultsContent.innerHTML = content;

    // Show Post Processing and Fine Tuning sections only when job has succeeded
    const postProcessingCard = document.getElementById("postProcessingCard");
    if (postProcessingCard) {
      postProcessingCard.style.display = status === "success" ? "block" : "none";
    }

    const fineTuningCard = document.getElementById("fineTuningCard");
    if (fineTuningCard) {
      fineTuningCard.style.display = status === "success" ? "block" : "none";
    }

    // Update fine-tuning layers from layer_cfg if available (for success status)
    if (status === "success" && layerCfg) {
      // Update use_torch_layers
      if (layerCfg.use_torch_layers !== undefined) {
        fineTuningUseTorchLayers = layerCfg.use_torch_layers;
        
        // Use setTimeout to ensure DOM is ready
        setTimeout(() => {
          const usePresetLayersCheckbox = document.getElementById("fineTuningUsePresetLayers");
          if (usePresetLayersCheckbox) {
            usePresetLayersCheckbox.checked = fineTuningUseTorchLayers;
          }
        }, 100);
      }

      // Update layers array
      if (Array.isArray(layerCfg.layers)) {
        // Deep copy the layers to avoid reference issues
        fineTuningLayers = layerCfg.layers.map((layer) => ({ ...layer }));
        
        // Initialize ftLayerDetails: all default layers are backbone with freeze: true
        // Set original_id to the index for default layers (static, never changes)
        ftLayerDetails = layerCfg.layers.map((_, index) => ({ type: "backbone", freeze: true, original_id: index }));
        
        // Use setTimeout to ensure DOM is ready
        setTimeout(() => {
          renderFineTuningLayers();
        }, 100);
      } else {
        fineTuningLayers = [];
        ftLayerDetails = [];
        setTimeout(() => {
          renderFineTuningLayers();
        }, 100);
      }
    }

    // Set up filter event listeners for validation error table if it exists
    if (originalValErrorTable.length > 0) {
      setupValErrorTableFilters();
    }
  }

  // Set up filter event listeners for validation error table
  function setupValErrorTableFilters() {
    const filterInputs = document.querySelectorAll(".test-error-filter-input");
    
    // Load existing filter values from currentValErrorFilters
    filterInputs.forEach((input) => {
      const column = input.getAttribute("data-column");
      if (currentValErrorFilters[column]) {
        input.value = currentValErrorFilters[column];
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
            delete currentValErrorFilters[column];
          } else {
            currentValErrorFilters[column] = filterValue;
          }
          updateValErrorTable();
        }, 300); // 300ms debounce
      });
    });
  }

  // Update validation error table with current filters
  function updateValErrorTable() {
    const wrapper = document.getElementById("test-error-table-wrapper");
    if (!wrapper || originalValErrorTable.length === 0) return;

    // Regenerate table with current filters
    const valErrorTableResult = generateValErrorTable(originalValErrorTable, currentValErrorFilters);
    wrapper.innerHTML = valErrorTableResult.html;

    // Re-setup filter listeners (since we replaced the HTML)
    setupValErrorTableFilters();
  }

  // Load and display experiments list
  function loadExperiments() {
    if (!experimentsList) return;

    // Show loading state
    experimentsList.innerHTML = '<p class="muted">Loading experiments...</p>';

    fetch("http://localhost:8000/history", {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    })
      .then(async (response) => {
        if (response.status === 500) {
          // Handle HttpException with status 500
          const errorData = await response.json();
          experimentsList.innerHTML = `
            <div class="experiment-error">
              <p style="color: #fecaca; margin-bottom: 0.5rem;">Error loading experiments:</p>
              <pre style="color: #d1d5db; font-size: 0.85rem; white-space: pre-wrap;">${JSON.stringify(errorData, null, 2)}</pre>
            </div>
          `;
          return;
        }

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        return response.json();
      })
      .then((data) => {
        if (!data) return; // Already handled error case

        // Check if we have the expected structure: { exps: List[Experiment] }
        if (data.exps && Array.isArray(data.exps)) {
          renderExperiments(data.exps);
        } else {
          experimentsList.innerHTML = '<p class="muted">No experiments found or unexpected response format.</p>';
        }
      })
      .catch((error) => {
        console.error("Failed to fetch experiments:", error);
        experimentsList.innerHTML = `
          <div class="experiment-error">
            <p style="color: #fecaca;">Failed to load experiments: ${error.message}</p>
          </div>
        `;
      });
  }

  // Render experiments list
  function renderExperiments(experiments) {
    if (!experimentsList) return;

    if (experiments.length === 0) {
      experimentsList.innerHTML = '<p class="muted">No experiments available.</p>';
      return;
    }

    experimentsList.innerHTML = "";

    experiments.forEach((experiment) => {
      const item = document.createElement("div");
      item.className = "experiment-item";

      const header = document.createElement("div");
      header.className = "experiment-item-header";

      const nameEl = document.createElement("div");
      nameEl.className = "experiment-name";
      nameEl.textContent = experiment.name || "(unnamed experiment)";
      nameEl.style.cursor = "pointer";
      nameEl.style.color = "#3b82f6";
      nameEl.style.textDecoration = "underline";
      nameEl.title = `Click to open: ${experiment.url || ""}`;

      // Make experiment name clickable to open URL in new tab
      nameEl.addEventListener("click", (e) => {
        e.preventDefault();
        if (experiment.url) {
          window.open(experiment.url, "_blank");
        } else {
          alert("No URL available for this experiment.");
        }
      });

      header.appendChild(nameEl);
      item.appendChild(header);

      experimentsList.appendChild(item);
    });
  }

  // Load experiments on page load
  loadExperiments();

  // Initialize Post Processing section
  renderPostProcessingList();
  setupPostProcessing();

  // Initialize Fine Tuning section
  populateFineTuningTransformSelect();
  renderFineTuningTransformList();
  renderFineTuningLayers();
  setupFineTuning();
});
