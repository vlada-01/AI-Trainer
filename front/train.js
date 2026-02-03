// Simple state for custom layers
let layers = [];

// Training configuration
let trainingConfig = {
  exp_name: "",
  run_name: "",
  model_name: "",
  log_train_metrics: false,
  train_cfg: {
    device: "cpu",
    epochs: 10,
    num_of_iters: 100,
    optimizer: { type: "", args: {} }, // OptimizerConfig
    lr_decay: null, // Optional LrDecay
    loss_fn: { type: "", args: {} }, // LossFnConfig
    metrics: [], // List of metrics
  },
};

function getDefaultLayer(type = "Linear") {
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

function renderLayers() {
  const container = document.getElementById("layersContainer");
  container.innerHTML = "";

  if (layers.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No layers added yet. Click “Add layer” to start.";
    container.appendChild(empty);
    return;
  }

  layers.forEach((layer, index) => {
    const card = document.createElement("div");
    card.className = "layer-card";
    card.draggable = true;
    card.dataset.index = String(index);

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
          <div class="layer-header-top-row">
            <span class="layer-index">Layer ${index + 1}</span>
          </div>
          <div class="layer-type-row">
            <label class="layer-type-label" for="layer-type-${index}">Layer type</label>
            <select id="layer-type-${index}" data-field="type" data-index="${index}">
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

function buildConfig() {
  const modelType = document.getElementById("modelType").value;
  const usePresetLayers = document.getElementById("usePresetLayers").checked;

  return {
    // snake_case keys for the backend
    model_type: modelType,
    use_torch_layers: usePresetLayers,
    layers: layers.map((layer) => ({ ...layer })),
  };
}

function updateModelConfigPreview() {
  const preview = document.getElementById("modelConfigPreview");
  if (!preview) return;
  const config = buildConfig();
  preview.textContent = JSON.stringify(config, null, 2);
}


function parseJsonSafely(jsonString, defaultValue = {}) {
  if (!jsonString || jsonString.trim() === "") {
    return defaultValue;
  }
  try {
    return JSON.parse(jsonString);
  } catch (e) {
    return null; // Return null to indicate parse error
  }
}

function buildTrainingConfig() {
  // Create a deep copy to avoid modifying the original
  const config = JSON.parse(JSON.stringify(trainingConfig));
  
  // Parse optimizer args if it's stored as string
  if (config.train_cfg.optimizer.args && typeof config.train_cfg.optimizer.args === 'string') {
    const parsed = parseJsonSafely(config.train_cfg.optimizer.args, {});
    config.train_cfg.optimizer.args = parsed !== null ? parsed : {};
  }
  
  // Parse loss_fn args if it's stored as string
  if (config.train_cfg.loss_fn.args && typeof config.train_cfg.loss_fn.args === 'string') {
    const parsed = parseJsonSafely(config.train_cfg.loss_fn.args, {});
    config.train_cfg.loss_fn.args = parsed !== null ? parsed : {};
  }
  
  // Handle optional lr_decay - if null or empty, don't include it
  if (config.train_cfg.lr_decay === null || 
      (config.train_cfg.lr_decay && config.train_cfg.lr_decay.type === "")) {
    delete config.train_cfg.lr_decay;
  } else if (config.train_cfg.lr_decay && typeof config.train_cfg.lr_decay.args === 'string') {
    // Parse lr_decay args if it's stored as string
    const parsed = parseJsonSafely(config.train_cfg.lr_decay.args, {});
    config.train_cfg.lr_decay.args = parsed !== null ? parsed : {};
  }
  
  return config;
}

function updateTrainingConfigPreview() {
  const preview = document.getElementById("trainingConfigPreview");
  if (!preview) return;
  const config = buildTrainingConfig();
  preview.textContent = JSON.stringify(config, null, 2);
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("train-form");
  const addLayerBtn = document.getElementById("addLayerBtn");
  const layersContainer = document.getElementById("layersContainer");
  const usePresetLayers = document.getElementById("usePresetLayers");
  let dragIndex = null;

  // Initial state
  layers = [getDefaultLayer("Conv2d"), getDefaultLayer("MaxPool2d"), getDefaultLayer("Linear")];
  renderLayers();
  updateModelConfigPreview();

  addLayerBtn.addEventListener("click", () => {
    layers.push(getDefaultLayer("Linear"));
    renderLayers();
  });

  // Drag & drop reordering
  layersContainer.addEventListener("dragstart", (event) => {
    const card = event.target.closest(".layer-card");
    if (!card) return;
    dragIndex = parseInt(card.dataset.index, 10);
    card.classList.add("dragging");
    event.dataTransfer.effectAllowed = "move";
  });

  layersContainer.addEventListener("dragend", (event) => {
    const card = event.target.closest(".layer-card");
    if (card) {
      card.classList.remove("dragging");
    }
    dragIndex = null;
    Array.from(layersContainer.children).forEach((child) =>
      child.classList.remove("drag-over")
    );
  });

  layersContainer.addEventListener("dragover", (event) => {
    event.preventDefault();
    const card = event.target.closest(".layer-card");
    if (!card || dragIndex === null) return;

    Array.from(layersContainer.children).forEach((child) =>
      child.classList.remove("drag-over")
    );
    card.classList.add("drag-over");
  });

  layersContainer.addEventListener("drop", (event) => {
    event.preventDefault();
    const targetCard = event.target.closest(".layer-card");
    if (!targetCard || dragIndex === null) return;

    const targetIndex = parseInt(targetCard.dataset.index, 10);
    if (Number.isNaN(targetIndex) || targetIndex === dragIndex) return;

    const [moved] = layers.splice(dragIndex, 1);
    layers.splice(targetIndex, 0, moved);

    dragIndex = null;
        renderLayers();
  });

  layersContainer.addEventListener("click", (event) => {
    const target = event.target;

    if (target.dataset.remove !== undefined) {
      const index = parseInt(target.dataset.remove, 10);
      layers.splice(index, 1);
      renderLayers();
      return;
    }

    if (target.dataset.move) {
      const index = parseInt(target.dataset.index, 10);
      if (target.dataset.move === "up" && index > 0) {
        [layers[index - 1], layers[index]] = [layers[index], layers[index - 1]];
      } else if (target.dataset.move === "down" && index < layers.length - 1) {
        [layers[index + 1], layers[index]] = [layers[index], layers[index + 1]];
      }
      renderLayers();
    }
  });

  layersContainer.addEventListener("change", (event) => {
    const target = event.target;
    const index = parseInt(target.dataset.index, 10);
    const field = target.dataset.field;

    if (Number.isNaN(index) || !field) return;

    if (field === "type") {
      // When changing type, reset to sensible defaults for that type
      layers[index] = getDefaultLayer(target.value);
        renderLayers();
      return;
    }

    if (target.type === "checkbox") {
      layers[index][field] = target.checked;
    } else if (target.type === "number") {
      const raw = target.value;
      // Allow optional fields (like stride for MaxPool2d) to be empty
      layers[index][field] = raw === "" ? null : Number(raw);
    } else {
      layers[index][field] = target.value;
    }
  });

  usePresetLayers.addEventListener("change", () => {});

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const config = buildConfig();
    console.log("Submitting model configuration:", config);

    fetch("http://localhost:8000/models/prepare_model", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(config),
    })
      .then((response) => response.json())
      .then((job) => {
        // Expecting job schema: { id, status, status_details, error, created_at, expires_at }
        if (!job || !job.id) {
          console.error("Unexpected model job response:", job);
          alert("Model job created, but response format was unexpected. Check console.");
          return;
        }

        // Add model job to local state (newest first) and start polling
        // Add local timestamp for sorting if created_at is missing
        if (!job.created_at) {
          job._addedAt = Date.now();
        }
        modelJobs.unshift(job);
        renderJobs();
        startModelJobPolling();
      })
      .catch((error) => {
        console.error("Failed to submit model configuration:", error);
        alert("Failed to submit model configuration. See console for details.");
      });
  });

  // Dataset form handling
  const datasetProvider = document.getElementById("datasetProvider");
  const sklearnConfigFields = document.getElementById("sklearnConfigFields");
  const hfConfigFields = document.getElementById("hfConfigFields");
  const batchSizeInput = document.getElementById("batchSize");
  const shuffleCheckbox = document.getElementById("shuffle");
  const datasetForm = document.getElementById("dataset-form");

  // Dataset configuration state
  let datasetConfigState = {
    provider: "sklearn",
    batchSize: 1,
    shuffle: false,
    sklearn: {
      dataset_fn: "",
      stratify: true,
      task: "classification",
      test_size: null,
      val_size: 0.2,
    },
    huggingface: {
      id: "",
      name: null,
      task: "classification",
      meta_type: "tabular",
      train_split: "",
      val_split: "",
      test_split: null,
      load_ds_args: {},
      mapper: {
        name: "simple",
        x_mapping: "",
        y_mapping: "",
      },
    },
    transforms: {
      train: { transform: [], target_transform: [] },
      valid: { transform: [], target_transform: [] },
      test: { transform: [], target_transform: [] },
    },
  };

  // Dataset jobs state (used only for status polling)
  let datasetJobs = [];
  let datasetJobPollInterval = null;
  const DATASET_JOB_POLL_INTERVAL_MS = 2000; // 2 seconds

  // Model jobs state (used only for status polling)
  let modelJobs = [];
  let modelJobPollInterval = null;
  const MODEL_JOB_POLL_INTERVAL_MS = 2000; // 2 seconds

  // Training jobs state (used only for status polling)
  let trainJobs = [];
  let trainJobPollInterval = null;
  const TRAIN_JOB_POLL_INTERVAL_MS = 2000; // 2 seconds

  // Helper: build dataset configuration payload
  function buildDatasetConfig() {
    // Build data_config with provider and provider-specific fields
    let data_config = {
      dataset_provider: datasetConfigState.provider,
    };

    if (datasetConfigState.provider === "sklearn") {
      data_config.dataset_fn = datasetConfigState.sklearn.dataset_fn;
      data_config.stratify = datasetConfigState.sklearn.stratify;
      data_config.task = datasetConfigState.sklearn.task;
      data_config.test_size = datasetConfigState.sklearn.test_size;
      data_config.val_size = datasetConfigState.sklearn.val_size;
    } else if (datasetConfigState.provider === "hugging face") {
      data_config.id = datasetConfigState.huggingface.id;
      data_config.name = datasetConfigState.huggingface.name;
      data_config.task = datasetConfigState.huggingface.task;
      data_config.meta_type = datasetConfigState.huggingface.meta_type;
      data_config.train_split = datasetConfigState.huggingface.train_split;
      data_config.val_split = datasetConfigState.huggingface.val_split;
      data_config.test_split = datasetConfigState.huggingface.test_split;
      data_config.load_ds_args = datasetConfigState.huggingface.load_ds_args;
      data_config.mapper = datasetConfigState.huggingface.mapper;
    }

    return {
      data_config: data_config,
      batch_size: datasetConfigState.batchSize,
      shuffle: datasetConfigState.shuffle,
      dataset_transforms: datasetConfigState.transforms,
    };
  }

  // Update provider fields visibility
  function updateProviderFields() {
    if (!datasetProvider) return;
    const provider = datasetProvider.value;
    datasetConfigState.provider = provider;

    if (provider === "hugging face") {
      if (sklearnConfigFields) {
        sklearnConfigFields.setAttribute("hidden", "");
      }
      if (hfConfigFields) {
        hfConfigFields.removeAttribute("hidden");
      }
    } else {
      if (sklearnConfigFields) {
        sklearnConfigFields.removeAttribute("hidden");
      }
      if (hfConfigFields) {
        hfConfigFields.setAttribute("hidden", "");
      }
    }
    
    // Update preview when provider changes
    updateDatasetConfigPreview();
  }

  if (datasetProvider) {
    datasetProvider.addEventListener("change", updateProviderFields);
    updateProviderFields();
  }

  // Batch size handling
  if (batchSizeInput) {
    batchSizeInput.addEventListener("input", () => {
      const value = parseInt(batchSizeInput.value, 10);
      if (!isNaN(value) && value >= 1) {
        datasetConfigState.batchSize = value;
      } else {
        datasetConfigState.batchSize = 1;
      }
    });
  }

  // Shuffle handling
  if (shuffleCheckbox) {
    shuffleCheckbox.addEventListener("change", () => {
      datasetConfigState.shuffle = shuffleCheckbox.checked;
    });
  }

  // Sklearn configuration fields
  const datasetFn = document.getElementById("datasetFn");
  const datasetStratify = document.getElementById("datasetStratify");
  const taskType = document.getElementById("taskType");
  const testSize = document.getElementById("testSize");
  const valSize = document.getElementById("valSize");

  if (datasetFn) {
    datasetFn.addEventListener("input", () => {
      datasetConfigState.sklearn.dataset_fn = datasetFn.value;
    });
  }

  if (datasetStratify) {
    datasetStratify.addEventListener("change", () => {
      datasetConfigState.sklearn.stratify = datasetStratify.checked;
    });
  }

  if (taskType) {
    taskType.addEventListener("change", () => {
      datasetConfigState.sklearn.task = taskType.value;
    });
  }

  if (testSize) {
    testSize.addEventListener("input", () => {
      const value = parseFloat(testSize.value);
      if (!isNaN(value) && value >= 0 && value <= 1) {
        datasetConfigState.sklearn.test_size = value;
      } else if (testSize.value === "" || testSize.value === null) {
        datasetConfigState.sklearn.test_size = null;
      }
    });
  }

  if (valSize) {
    valSize.addEventListener("input", () => {
      const value = parseFloat(valSize.value);
      if (!isNaN(value) && value >= 0 && value <= 1) {
        datasetConfigState.sklearn.val_size = value;
      } else {
        datasetConfigState.sklearn.val_size = 0.2;
      }
    });
  }

  // HuggingFace configuration fields
  const hfId = document.getElementById("hfId");
  const hfName = document.getElementById("hfName");
  const hfTask = document.getElementById("hfTask");
  const hfMetaType = document.getElementById("hfMetaType");
  const hfTrainSplit = document.getElementById("hfTrainSplit");
  const hfValSplit = document.getElementById("hfValSplit");
  const hfTestSplit = document.getElementById("hfTestSplit");
  const hfLoadDsArgs = document.getElementById("hfLoadDsArgs");
  const hfLoadDsArgsError = document.getElementById("hfLoadDsArgsError");
  const hfMapper = document.getElementById("hfMapper");
  const simpleMapperXMapping = document.getElementById("simpleMapperXMapping");
  const simpleMapperYMapping = document.getElementById("simpleMapperYMapping");
  const simpleMapperFields = document.getElementById("simpleMapperFields");
  const hfDatasetInfoBtn = document.getElementById("hfDatasetInfoBtn");

  // Helper function to safely parse JSON
  function parseJsonSafely(jsonString, defaultValue) {
    try {
      return JSON.parse(jsonString);
    } catch (e) {
      return defaultValue;
    }
  }

  if (hfId) {
    hfId.addEventListener("input", () => {
      datasetConfigState.huggingface.id = hfId.value;
    });
  }

  if (hfName) {
    hfName.addEventListener("input", () => {
      const value = hfName.value.trim();
      datasetConfigState.huggingface.name = value === "" ? null : value;
    });
  }

  if (hfTask) {
    hfTask.addEventListener("change", () => {
      datasetConfigState.huggingface.task = hfTask.value;
    });
  }

  if (hfMetaType) {
    hfMetaType.addEventListener("change", () => {
      datasetConfigState.huggingface.meta_type = hfMetaType.value;
    });
  }

  if (hfTrainSplit) {
    hfTrainSplit.addEventListener("input", () => {
      datasetConfigState.huggingface.train_split = hfTrainSplit.value;
    });
  }

  if (hfValSplit) {
    hfValSplit.addEventListener("input", () => {
      datasetConfigState.huggingface.val_split = hfValSplit.value;
    });
  }

  if (hfTestSplit) {
    hfTestSplit.addEventListener("input", () => {
      const value = hfTestSplit.value.trim();
      datasetConfigState.huggingface.test_split = value === "" ? null : value;
    });
  }

  if (hfLoadDsArgs) {
    hfLoadDsArgs.addEventListener("input", () => {
      const jsonString = hfLoadDsArgs.value.trim();
      if (jsonString === "") {
        datasetConfigState.huggingface.load_ds_args = {};
        if (hfLoadDsArgsError) {
          hfLoadDsArgsError.style.display = "none";
          hfLoadDsArgsError.textContent = "";
        }
      } else {
        const parsed = parseJsonSafely(jsonString, null);
        if (parsed === null) {
          // Invalid JSON - show error
          if (hfLoadDsArgsError) {
            hfLoadDsArgsError.style.display = "block";
            hfLoadDsArgsError.textContent = "Invalid JSON format";
          }
        } else {
          datasetConfigState.huggingface.load_ds_args = parsed;
          if (hfLoadDsArgsError) {
            hfLoadDsArgsError.style.display = "none";
            hfLoadDsArgsError.textContent = "";
          }
        }
      }
    });
  }

  if (hfMapper) {
    hfMapper.addEventListener("change", () => {
      datasetConfigState.huggingface.mapper.name = hfMapper.value;
      // Show/hide mapper-specific fields based on selection
      if (hfMapper.value === "simple" && simpleMapperFields) {
        simpleMapperFields.removeAttribute("hidden");
      } else if (simpleMapperFields) {
        simpleMapperFields.setAttribute("hidden", "");
      }
    });
    // Initialize mapper fields visibility
    if (hfMapper.value === "simple" && simpleMapperFields) {
      simpleMapperFields.removeAttribute("hidden");
    }
  }

  if (simpleMapperXMapping) {
    simpleMapperXMapping.addEventListener("input", () => {
      datasetConfigState.huggingface.mapper.x_mapping = simpleMapperXMapping.value;
    });
  }

  if (simpleMapperYMapping) {
    simpleMapperYMapping.addEventListener("input", () => {
      datasetConfigState.huggingface.mapper.y_mapping = simpleMapperYMapping.value;
    });
  }

  // HuggingFace dataset info lookup
  if (hfDatasetInfoBtn) {
    hfDatasetInfoBtn.addEventListener("click", () => {
      // Ensure HuggingFace provider is selected
      if (datasetConfigState.provider !== "hugging face") {
        alert("Please select the HuggingFace provider first.");
        return;
      }

      const datasetId = datasetConfigState.huggingface.id;
      if (!datasetId || datasetId.trim() === "") {
        alert("Please enter a Dataset ID before requesting dataset info.");
        return;
      }

      const url = "http://localhost:8000/data/get_dataset_info";
      const payload = {
        dataset_provider: datasetConfigState.provider,
        id: datasetId.trim(),
        name: datasetConfigState.huggingface.name,
      };

      // Expose payload for debugging
      console.log("Dataset info request payload:", payload);

      // Open a new blank tab immediately to avoid popup blockers
      const infoWindow = window.open("", "_blank");
      if (!infoWindow) {
        alert("Unable to open a new tab. Please allow popups for this site.");
        return;
      }

      // Basic loading skeleton
      infoWindow.document.write(
        "<!DOCTYPE html><html><head><title>Dataset info</title></head><body>" +
          "<h1>Dataset info</h1>" +
          "<h2>Request payload</h2>" +
          "<pre id='datasetRequest'></pre>" +
          "<h2>Response</h2>" +
          "<pre id='datasetInfo'>Loading dataset info...</pre>" +
        "</body></html>"
      );
      infoWindow.document.close();

      // Show the request JSON in the new tab
      const requestPre = infoWindow.document.getElementById("datasetRequest");
      if (requestPre) {
        requestPre.textContent = JSON.stringify(payload, null, 2);
      }

      fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
      })
        .then(async (response) => {
          const text = await response.text();
          let formatted = text;

          // Try to pretty-print JSON if possible
          try {
            const parsed = JSON.parse(text);
            formatted = JSON.stringify(parsed, null, 2);
          } catch (_) {
            // Not valid JSON, keep raw text
          }

          const pre = infoWindow.document.getElementById("datasetInfo");
          if (pre) {
            pre.textContent = formatted;
          } else if (infoWindow.document && infoWindow.document.body) {
            const newPre = infoWindow.document.createElement("pre");
            newPre.textContent = formatted;
            infoWindow.document.body.innerHTML = "";
            infoWindow.document.body.appendChild(newPre);
          }
        })
        .catch((error) => {
          const pre = infoWindow.document.getElementById("datasetInfo");
          const message = `Error fetching dataset info: ${error}`;
          if (pre) {
            pre.textContent = message;
          } else if (infoWindow.document && infoWindow.document.body) {
            const newPre = infoWindow.document.createElement("pre");
            newPre.textContent = message;
            infoWindow.document.body.innerHTML = "";
            infoWindow.document.body.appendChild(newPre);
          }
        });
    });
  }

  // ----- Dataset jobs rendering and polling -----

  function isTerminalJobStatus(status) {
    if (!status) return false;
    const s = status.toString().toLowerCase();
    return s === "success" || s === "failed" || s === "error";
  }

  function startDatasetJobPolling() {
    if (datasetJobPollInterval !== null) return;

    datasetJobPollInterval = setInterval(() => {
      const activeJobs = datasetJobs.filter((job) => !isTerminalJobStatus(job.status));
      if (activeJobs.length === 0) {
        clearInterval(datasetJobPollInterval);
        datasetJobPollInterval = null;
        return;
      }

      activeJobs.forEach((job) => {
        fetch(`http://localhost:8000/data/dataset_status/${job.id}`, {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
        })
          .then((response) => response.json())
          .then((updated) => {
            // Find and update the job in local state
            const idx = datasetJobs.findIndex((j) => j.id === job.id);
            if (idx !== -1) {
              datasetJobs[idx] = {
                ...datasetJobs[idx],
                ...updated,
              };
              renderJobs();
            }
          })
          .catch((error) => {
            console.error("Failed to fetch dataset job status:", error);
          });
      });
    }, DATASET_JOB_POLL_INTERVAL_MS);
  }

  function startModelJobPolling() {
    if (modelJobPollInterval !== null) return;

    modelJobPollInterval = setInterval(() => {
      const activeJobs = modelJobs.filter((job) => !isTerminalJobStatus(job.status));
      if (activeJobs.length === 0) {
        clearInterval(modelJobPollInterval);
        modelJobPollInterval = null;
        return;
      }

      activeJobs.forEach((job) => {
        fetch(`http://localhost:8000/models/model_status/${job.id}`, {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
        })
          .then((response) => response.json())
          .then((updated) => {
            // Find and update the job in local state
            const idx = modelJobs.findIndex((j) => j.id === job.id);
            if (idx !== -1) {
              modelJobs[idx] = {
                ...modelJobs[idx],
                ...updated,
              };
              renderJobs();
            }
          })
          .catch((error) => {
            console.error("Failed to fetch model job status:", error);
          });
      });
    }, MODEL_JOB_POLL_INTERVAL_MS);
  }

  function startTrainJobPolling() {
    if (trainJobPollInterval !== null) return;

    trainJobPollInterval = setInterval(() => {
      const activeJobs = trainJobs.filter((job) => !isTerminalJobStatus(job.status));
      if (activeJobs.length === 0) {
        clearInterval(trainJobPollInterval);
        trainJobPollInterval = null;
        return;
      }

      activeJobs.forEach((job) => {
        fetch(`http://localhost:8000/train/train_status/${job.id}`, {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
        })
          .then((response) => response.json())
          .then((updated) => {
            // Find and update the job in local state
            const idx = trainJobs.findIndex((j) => j.id === job.id);
            if (idx !== -1) {
              trainJobs[idx] = {
                ...trainJobs[idx],
                ...updated,
              };
              renderJobs();
            }
          })
          .catch((error) => {
            console.error("Failed to fetch training job status:", error);
          });
      });
    }, TRAIN_JOB_POLL_INTERVAL_MS);
  }

  // Render training jobs list
  function renderJobs() {
    if (!trainingJobsList) return;

    trainingJobsList.innerHTML = "";

    const combinedJobs = [
      ...datasetJobs.map((job) => ({ ...job, _type: "Dataset" })),
      ...modelJobs.map((job) => ({ ...job, _type: "Model" })),
      ...trainJobs.map((job) => ({ ...job, _type: "Training" })),
    ];

    // Sort by creation time (newest first)
    // Prefer created_at from server, fall back to _addedAt local timestamp
    combinedJobs.sort((a, b) => {
      const timeA = a.created_at 
        ? new Date(a.created_at).getTime() 
        : (a._addedAt || 0);
      const timeB = b.created_at 
        ? new Date(b.created_at).getTime() 
        : (b._addedAt || 0);
      // Sort descending (newest first)
      return timeB - timeA;
    });

    if (combinedJobs.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No jobs submitted yet.";
      trainingJobsList.appendChild(empty);
      return;
    }

    combinedJobs.forEach((job) => {
      const item = document.createElement("div");
      item.className = "training-job-item";

      const header = document.createElement("div");
      header.className = "training-job-header";

      const headerLeft = document.createElement("div");
      headerLeft.className = "training-job-header-left";

      const typeEl = document.createElement("div");
      typeEl.className = "training-job-type";
      typeEl.textContent = job._type || "Job";

      const idEl = document.createElement("div");
      idEl.className = "training-job-id";
      idEl.textContent = `Id: ${job.id || "(no id)"}`;

      headerLeft.appendChild(typeEl);
      headerLeft.appendChild(idEl);

      const statusEl = document.createElement("span");
      statusEl.className = "training-job-status";
      const status = (job.status || "").toString().toLowerCase().trim();
      
      // Apply status classes to both the status badge and the item background
      if (status === "success" || status === "completed") {
        statusEl.classList.add("status-success");
        item.classList.add("status-success");
      } else if (status === "failed" || status === "error" || status === "failure") {
        statusEl.classList.add("status-failed");
        item.classList.add("status-failed");
      } else if (status === "pending" || status === "queued" || status === "waiting") {
        statusEl.classList.add("status-pending");
        item.classList.add("status-pending");
      } else if (status === "in_progress" || status === "running" || status === "processing" || status === "active") {
        statusEl.classList.add("status-in_progress");
        item.classList.add("status-in_progress");
      } else {
        // Default: apply pending style for unknown statuses
        item.classList.add("status-pending");
      }
      statusEl.textContent = job.status || "unknown";

      header.appendChild(headerLeft);
      header.appendChild(statusEl);
      item.appendChild(header);

      if (job.created_at) {
        const created = document.createElement("div");
        created.className = "training-job-timestamp";
        created.textContent = `Created: ${job.created_at}`;
        item.appendChild(created);
      }

      if (job.expires_at) {
        const expires = document.createElement("div");
        expires.className = "training-job-timestamp";
        expires.textContent = `Expires: ${job.expires_at}`;
        item.appendChild(expires);
      }

      // Always show Details section
      const details = document.createElement("div");
      details.className = "training-job-details";
      
      if (job.status_details) {
        // Check if status_details is an object/dictionary
        if (typeof job.status_details === "object" && job.status_details !== null) {
          // Pretty-print as JSON
          try {
            details.textContent = `Details: ${JSON.stringify(job.status_details, null, 2)}`;
          } catch (e) {
            details.textContent = `Details: ${String(job.status_details)}`;
          }
        } else {
          // It's a string or other primitive
          details.textContent = `Details: ${job.status_details}`;
        }
      } else {
        details.textContent = "Details: (none)";
      }
      item.appendChild(details);

      // Always show Error section
      const error = document.createElement("div");
      error.className = "training-job-error";

      if (job.error) {
        if (typeof job.error === "object" && job.error !== null) {
          // Pretty-print object error
          try {
            error.textContent = `Error: ${JSON.stringify(job.error, null, 2)}`;
          } catch (e) {
            error.textContent = `Error: ${String(job.error)}`;
          }
        } else {
          error.textContent = `Error: ${job.error}`;
        }
      } else {
        error.textContent = "Error: (none)";
      }
      item.appendChild(error);

      trainingJobsList.appendChild(item);
    });
  }

  // Transform handling
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

  function renderTransformList(split, type) {
    let listId;
    if (type === "target_transform") {
      listId = `${split}TargetTransformList`;
    } else {
      listId = `${split}TransformList`;
    }
    const container = document.getElementById(listId);
    if (!container) return;

    const transforms = datasetConfigState.transforms[split][type];
    container.innerHTML = "";

    if (transforms.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No transformations added yet.";
      container.appendChild(empty);
      return;
    }

    transforms.forEach((transformObj, index) => {
      const item = document.createElement("div");
      item.className = "transform-item";
      item.draggable = true;
      item.dataset.index = index;
      item.dataset.split = split;
      item.dataset.type = type;

      const transformName = transformObj.name;
      const displayName = transformDisplayNames[transformName] || transformName.replace(/_/g, " ");

      let innerHTML = `
        <span class="drag-handle">⋮⋮</span>
        <span class="transform-name">${displayName}</span>
      `;

      if (transformName === "normalize_l1_l2") {
        const pValue = transformObj.value && transformObj.value.p !== undefined ? transformObj.value.p : 1;
        innerHTML += `
          <label style="margin-left: 10px; display: inline-flex; align-items: center; gap: 5px;">
            p:
            <select class="l1l2-p-select" data-split="${split}" data-type="${type}" data-index="${index}" style="padding: 2px 5px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--bg-secondary); color: var(--text-primary);">
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
            <input type="number" step="0.01" min="0" max="1" class="transform-value-input" data-split="${split}" data-type="${type}" data-index="${index}" value="${pValue}" style="width: 60px; padding: 2px 5px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--bg-secondary); color: var(--text-primary);" />
          </label>
        `;
      } else if (transformName === "random_vertical_flip") {
        const pValue = transformObj.value && transformObj.value.p !== undefined ? transformObj.value.p : 0.5;
        innerHTML += `
          <label style="margin-left: 10px; display: inline-flex; align-items: center; gap: 5px;">
            p:
            <input type="number" step="0.01" min="0" max="1" class="transform-value-input" data-split="${split}" data-type="${type}" data-index="${index}" value="${pValue}" style="width: 60px; padding: 2px 5px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--bg-secondary); color: var(--text-primary);" />
          </label>
        `;
      } else if (transformName === "random_rotation") {
        const alphaValue = transformObj.value && transformObj.value.alpha !== undefined ? transformObj.value.alpha : 15;
        innerHTML += `
          <label style="margin-left: 10px; display: inline-flex; align-items: center; gap: 5px;">
            alpha:
            <input type="number" step="1" min="0" class="transform-value-input" data-split="${split}" data-type="${type}" data-index="${index}" value="${alphaValue}" style="width: 60px; padding: 2px 5px; border-radius: 4px; border: 1px solid var(--border-color); background: var(--bg-secondary); color: var(--text-primary);" />
          </label>
        `;
      }

      innerHTML += `<button type="button" class="btn small secondary" data-remove="${index}">Remove</button>`;
      item.innerHTML = innerHTML;
      container.appendChild(item);
    });
  }

  function addTransform(split, type, transformName) {
    if (!transformName) return;

    const exists = datasetConfigState.transforms[split][type].some(t => t.name === transformName);
    if (exists) {
      alert(`"${transformName}" is already in the list.`);
      return;
    }

    if (transformName === "normalize_l1_l2") {
      datasetConfigState.transforms[split][type].push({ name: "normalize_l1_l2", value: { p: 1 } });
    } else if (transformName === "img_to_tensor") {
      datasetConfigState.transforms[split][type].push({ name: "img_to_tensor", value: true });
    } else if (transformName === "random_horizontal_flip") {
      datasetConfigState.transforms[split][type].push({ name: "random_horizontal_flip", value: { p: 0.5 } });
    } else if (transformName === "random_vertical_flip") {
      datasetConfigState.transforms[split][type].push({ name: "random_vertical_flip", value: { p: 0.5 } });
    } else if (transformName === "random_rotation") {
      datasetConfigState.transforms[split][type].push({ name: "random_rotation", value: { alpha: 15 } });
    } else {
      datasetConfigState.transforms[split][type].push({ name: transformName, value: true });
    }

    renderTransformList(split, type);
  }

  function removeTransform(split, type, index) {
    datasetConfigState.transforms[split][type].splice(index, 1);
    renderTransformList(split, type);
  }

  // Setup transform toggles
  function setupTransformToggles() {
    document.querySelectorAll(".transform-toggle").forEach((toggle) => {
      if (toggle.dataset.hasListener === "true") return;
      toggle.dataset.hasListener = "true";
      toggle.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        const split = toggle.dataset.split;
        const content = document.getElementById(`${split}TransformContent`);
        const icon = toggle.querySelector(".toggle-icon");

        if (content) {
          const isHidden = content.hidden;
          content.hidden = !isHidden;
          icon.textContent = isHidden ? "▼" : "▶";
        }
      });
    });
  }

  // Setup transform select dropdowns
  function setupTransformSelects() {
    document.querySelectorAll(".transform-select").forEach((select) => {
      if (select.dataset.hasListener === "true") return;
      select.dataset.hasListener = "true";
      select.addEventListener("change", (e) => {
        if (!e.target.value) return;
        const listId = e.target.id;
        let split, type;

        if (listId.includes("TransformSelect") && !listId.includes("Target")) {
          split = listId.replace("TransformSelect", "");
          type = "transform";
        } else if (listId.includes("TargetTransformSelect")) {
          split = listId.replace("TargetTransformSelect", "");
          type = "target_transform";
        }

        if (split && type) {
          addTransform(split, type, e.target.value);
          e.target.value = "";
        }
      });
    });
  }

  // Setup transform list interactions
  function setupTransformLists() {
    document.querySelectorAll(".transform-list").forEach((list) => {
      // Remove button handling
      list.addEventListener("click", (e) => {
        if (e.target.dataset.remove !== undefined) {
          const item = e.target.closest(".transform-item");
          const split = item.dataset.split;
          const type = item.dataset.type;
          const index = parseInt(e.target.dataset.remove, 10);
          removeTransform(split, type, index);
        }
      });

      // Parameter changes for transforms
      list.addEventListener("change", (e) => {
        if (e.target.classList.contains("l1l2-p-select")) {
          const split = e.target.dataset.split;
          const type = e.target.dataset.type;
          const index = parseInt(e.target.dataset.index, 10);
          const pValue = parseInt(e.target.value, 10);

          const transform = datasetConfigState.transforms[split][type][index];
          if (transform && transform.name === "normalize_l1_l2") {
            transform.value = { p: pValue };
          }
        } else if (e.target.classList.contains("transform-value-input")) {
          const split = e.target.dataset.split;
          const type = e.target.dataset.type;
          const index = parseInt(e.target.dataset.index, 10);
          const inputValue = parseFloat(e.target.value);

          const transform = datasetConfigState.transforms[split][type][index];
          if (transform) {
            if (transform.name === "random_horizontal_flip" || transform.name === "random_vertical_flip") {
              transform.value = { p: inputValue };
            } else if (transform.name === "random_rotation") {
              transform.value = { alpha: parseInt(e.target.value, 10) };
            }
          }
        }
      });
      
      // Handle input events for real-time updates (for number inputs)
      list.addEventListener("input", (e) => {
        if (e.target.classList.contains("transform-value-input")) {
          const split = e.target.dataset.split;
          const type = e.target.dataset.type;
          const index = parseInt(e.target.dataset.index, 10);
          const inputValue = parseFloat(e.target.value);

          const transform = datasetConfigState.transforms[split][type][index];
          if (transform) {
            if (transform.name === "random_horizontal_flip" || transform.name === "random_vertical_flip") {
              transform.value = { p: inputValue };
            } else if (transform.name === "random_rotation") {
              transform.value = { alpha: parseInt(e.target.value, 10) };
            }
          }
        }
      });
    });
  }

  // Initialize transform functionality
  setupTransformToggles();
  setupTransformSelects();
  setupTransformLists();

  // Dataset form submission
  if (datasetForm) {
    datasetForm.addEventListener("submit", (event) => {
      event.preventDefault();

      const payload = buildDatasetConfig();
      console.log("Submitting dataset configuration:", payload);

      fetch("http://localhost:8000/data/prepare_dataset", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
      })
        .then((response) => response.json())
        .then((job) => {
          // Expecting job schema:
          // { id, status, status_details, error, created_at, expires_at }
          if (!job || !job.id) {
            console.error("Unexpected job response:", job);
            alert("Dataset job created, but response format was unexpected. Check console.");
            return;
          }

          // Add job to local state (newest first) for status polling
          // Add local timestamp for sorting if created_at is missing
          if (!job.created_at) {
            job._addedAt = Date.now();
          }
          datasetJobs.unshift(job);
          renderJobs();
          startDatasetJobPolling();
        })
        .catch((error) => {
          console.error("Failed to submit dataset configuration:", error);
          alert("Failed to submit dataset configuration. See console for details.");
        });
    });
  }

  // Toggle between dataset preparation form and configuration preview
  const datasetPreparationCard = document.getElementById("datasetPreparationCard");
  const datasetConfigCard = document.getElementById("datasetConfigCard");
  const toggleDatasetConfigBtn = document.getElementById("toggleDatasetConfigBtn");
  const toggleDatasetConfigBtnBack = document.getElementById("toggleDatasetConfigBtnBack");

  function updateDatasetConfigPreview() {
    const preview = document.getElementById("datasetConfigPreview");
    if (!preview) return;

    const config = buildDatasetConfig();
    preview.textContent = JSON.stringify(config, null, 2);
  }

  if (toggleDatasetConfigBtn && datasetPreparationCard && datasetConfigCard) {
    toggleDatasetConfigBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      datasetPreparationCard.setAttribute("hidden", "");
      datasetConfigCard.removeAttribute("hidden");
      updateDatasetConfigPreview();
    });
  }

  if (toggleDatasetConfigBtnBack && datasetPreparationCard && datasetConfigCard) {
    toggleDatasetConfigBtnBack.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      datasetConfigCard.setAttribute("hidden", "");
      datasetPreparationCard.removeAttribute("hidden");
    });
  }

  // Ensure datasetConfigCard starts hidden on page load
  if (datasetConfigCard) {
    datasetConfigCard.setAttribute("hidden", "");
  }

  // Toggle between model preparation form and configuration preview
  const modelPreparationCard = document.getElementById("modelPreparationCard");
  const modelConfigCard = document.getElementById("modelConfigCard");
  const toggleModelConfigBtn = document.getElementById("toggleModelConfigBtn");
  const toggleModelConfigBtnBack = document.getElementById("toggleModelConfigBtnBack");

  if (toggleModelConfigBtn && modelPreparationCard && modelConfigCard) {
    toggleModelConfigBtn.addEventListener("click", () => {
      modelPreparationCard.hidden = true;
      modelConfigCard.hidden = false;
      updateModelConfigPreview(); // Update the preview with current config
    });
  }

  if (toggleModelConfigBtnBack && modelPreparationCard && modelConfigCard) {
    toggleModelConfigBtnBack.addEventListener("click", () => {
      modelConfigCard.hidden = true;
      modelPreparationCard.hidden = false;
    });
  }


  // Training form handling
  const trainingForm = document.getElementById("training-form");
  const expName = document.getElementById("expName");
  const runName = document.getElementById("runName");
  const modelName = document.getElementById("modelName");
  const logTrainMetrics = document.getElementById("logTrainMetrics");
  const trainingParametersCard = document.getElementById("trainingParametersCard");
  const trainingConfigCard = document.getElementById("trainingConfigCard");
  const toggleTrainingConfigBtn = document.getElementById("toggleTrainingConfigBtn");
  const toggleTrainingConfigBtnBack = document.getElementById("toggleTrainingConfigBtnBack");

  if (expName) {
    expName.addEventListener("input", () => {
      trainingConfig.exp_name = expName.value;
      updateTrainingConfigPreview();
    });
  }

  if (runName) {
    runName.addEventListener("input", () => {
      trainingConfig.run_name = runName.value;
      updateTrainingConfigPreview();
    });
  }

  if (modelName) {
    modelName.addEventListener("input", () => {
      trainingConfig.model_name = modelName.value;
      updateTrainingConfigPreview();
    });
  }

  if (logTrainMetrics) {
    logTrainMetrics.addEventListener("change", () => {
      trainingConfig.log_train_metrics = logTrainMetrics.checked;
      updateTrainingConfigPreview();
    });
  }

  // Train config fields
  const device = document.getElementById("device");
  const epochs = document.getElementById("epochs");
  const numOfIters = document.getElementById("numOfIters");
  const metricsSelect = document.getElementById("metricsSelect");
  const metricsList = document.getElementById("metricsList");
  const trainingJobsList = document.getElementById("trainingJobsList");

  if (device) {
    device.addEventListener("change", () => {
      trainingConfig.train_cfg.device = device.value;
      updateTrainingConfigPreview();
    });
  }

  if (epochs) {
    epochs.addEventListener("input", () => {
      const value = parseInt(epochs.value, 10);
      if (!isNaN(value) && value >= 1) {
        trainingConfig.train_cfg.epochs = value;
      } else {
        trainingConfig.train_cfg.epochs = 10;
      }
      updateTrainingConfigPreview();
    });
  }

  if (numOfIters) {
    numOfIters.addEventListener("input", () => {
      const value = parseInt(numOfIters.value, 10);
      if (!isNaN(value) && value >= 1) {
        trainingConfig.train_cfg.num_of_iters = value;
      } else {
        trainingConfig.train_cfg.num_of_iters = 100;
      }
      updateTrainingConfigPreview();
    });
  }

  // Metrics list handling (similar to transforms)
  function renderMetricsList() {
    if (!metricsList) return;
    metricsList.innerHTML = "";
    
    if (trainingConfig.train_cfg.metrics.length === 0) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No metrics added yet.";
      metricsList.appendChild(empty);
      return;
    }

    trainingConfig.train_cfg.metrics.forEach((metric, index) => {
      const item = document.createElement("div");
      item.className = "transform-item";
      // Capitalize first letter, keep rest as-is
      const displayMetric = metric.charAt(0).toUpperCase() + metric.slice(1);
      item.innerHTML = `
        <span class="transform-name">${displayMetric}</span>
        <button type="button" class="btn small secondary" data-remove="${index}">Remove</button>
      `;
      metricsList.appendChild(item);
    });
  }

  if (metricsSelect) {
    metricsSelect.addEventListener("change", (e) => {
      if (!e.target.value) return;
      const metric = e.target.value;
      // Capitalize first letter, keep rest as-is
      const capitalizedMetric = metric.charAt(0).toUpperCase() + metric.slice(1);
      if (trainingConfig.train_cfg.metrics.includes(capitalizedMetric)) {
        alert(`"${capitalizedMetric}" is already in the list.`);
        return;
      }
      trainingConfig.train_cfg.metrics.push(capitalizedMetric);
      renderMetricsList();
      updateTrainingConfigPreview();
      e.target.value = ""; // Reset dropdown
    });
  }

  if (metricsList) {
    metricsList.addEventListener("click", (e) => {
      if (e.target.dataset.remove !== undefined) {
        const index = parseInt(e.target.dataset.remove, 10);
        trainingConfig.train_cfg.metrics.splice(index, 1);
        renderMetricsList();
        updateTrainingConfigPreview();
      }
    });
  }

  // Initialize metrics list
  renderMetricsList();

  // Toggle collapsible config sections (optimizer, lrDecay, lossFn)
  document.querySelectorAll(".transform-toggle[data-config]").forEach((toggle) => {
    toggle.addEventListener("click", () => {
      const configType = toggle.dataset.config;
      let content;
      if (configType === "optimizer") {
        content = document.getElementById("optimizerContent");
      } else if (configType === "lrDecay") {
        content = document.getElementById("lrDecayContent");
      } else if (configType === "lossFn") {
        content = document.getElementById("lossFnContent");
      }
      const icon = toggle.querySelector(".toggle-icon");
      
      if (content) {
        const isHidden = content.hidden;
        content.hidden = !isHidden;
        icon.textContent = isHidden ? "▼" : "▶";
      }
    });
  });

  // Optimizer, LrDecay, and LossFn configuration
  const optimizerType = document.getElementById("optimizerType");
  const optimizerArgs = document.getElementById("optimizerArgs");
  const optimizerArgsError = document.getElementById("optimizerArgsError");
  const lrDecayType = document.getElementById("lrDecayType");
  const lrDecayArgs = document.getElementById("lrDecayArgs");
  const lrDecayArgsError = document.getElementById("lrDecayArgsError");
  const lossFnType = document.getElementById("lossFnType");
  const lossFnArgs = document.getElementById("lossFnArgs");
  const lossFnArgsError = document.getElementById("lossFnArgsError");

  function validateJsonInput(textarea, errorElement, configPath) {
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

  function updateConfigFromJson(textarea, configPath, errorElement) {
    const value = textarea.value.trim();
    if (value === "") {
      // Set to empty object if empty
      const keys = configPath.split('.');
      let obj = trainingConfig.train_cfg;
      for (let i = 0; i < keys.length - 1; i++) {
        obj = obj[keys[i]];
      }
      obj[keys[keys.length - 1]] = {};
      updateTrainingConfigPreview();
      return;
    }
    
    if (validateJsonInput(textarea, errorElement, configPath)) {
      try {
        const parsed = JSON.parse(value);
        const keys = configPath.split('.');
        let obj = trainingConfig.train_cfg;
        for (let i = 0; i < keys.length - 1; i++) {
          obj = obj[keys[i]];
        }
        obj[keys[keys.length - 1]] = parsed;
        updateTrainingConfigPreview();
      } catch (e) {
        // Already handled by validateJsonInput
      }
    }
  }

  // Optimizer handlers
  if (optimizerType) {
    optimizerType.addEventListener("input", () => {
      trainingConfig.train_cfg.optimizer.type = optimizerType.value;
      updateTrainingConfigPreview();
    });
  }

  if (optimizerArgs) {
    optimizerArgs.addEventListener("input", () => {
      updateConfigFromJson(optimizerArgs, "optimizer.args", optimizerArgsError);
    });
    optimizerArgs.addEventListener("blur", () => {
      validateJsonInput(optimizerArgs, optimizerArgsError, "optimizer.args");
    });
  }

  // LrDecay handlers
  if (lrDecayType) {
    lrDecayType.addEventListener("input", () => {
      if (lrDecayType.value.trim() === "") {
        trainingConfig.train_cfg.lr_decay = null;
      } else {
        if (!trainingConfig.train_cfg.lr_decay) {
          trainingConfig.train_cfg.lr_decay = { type: "", args: {} };
        }
        trainingConfig.train_cfg.lr_decay.type = lrDecayType.value;
      }
      updateTrainingConfigPreview();
    });
  }

  if (lrDecayArgs) {
    lrDecayArgs.addEventListener("input", () => {
      if (lrDecayType && lrDecayType.value.trim() !== "") {
        if (!trainingConfig.train_cfg.lr_decay) {
          trainingConfig.train_cfg.lr_decay = { type: lrDecayType.value, args: {} };
        }
        updateConfigFromJson(lrDecayArgs, "lr_decay.args", lrDecayArgsError);
      }
    });
    lrDecayArgs.addEventListener("blur", () => {
      validateJsonInput(lrDecayArgs, lrDecayArgsError, "lr_decay.args");
    });
  }

  // LossFn handlers
  if (lossFnType) {
    lossFnType.addEventListener("input", () => {
      trainingConfig.train_cfg.loss_fn.type = lossFnType.value;
      updateTrainingConfigPreview();
    });
  }

  if (lossFnArgs) {
    lossFnArgs.addEventListener("input", () => {
      updateConfigFromJson(lossFnArgs, "loss_fn.args", lossFnArgsError);
    });
    lossFnArgs.addEventListener("blur", () => {
      validateJsonInput(lossFnArgs, lossFnArgsError, "loss_fn.args");
    });
  }

  // Toggle between training parameters form and configuration preview
  if (toggleTrainingConfigBtn && trainingParametersCard && trainingConfigCard) {
    toggleTrainingConfigBtn.addEventListener("click", () => {
      trainingParametersCard.hidden = true;
      trainingConfigCard.hidden = false;
      updateTrainingConfigPreview();
    });
  }

  if (toggleTrainingConfigBtnBack && trainingParametersCard && trainingConfigCard) {
    toggleTrainingConfigBtnBack.addEventListener("click", () => {
      trainingConfigCard.hidden = true;
      trainingParametersCard.hidden = false;
    });
  }

  if (trainingForm) {
    trainingForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const config = buildTrainingConfig();
      console.log("Submitting training configuration:", config);

      fetch("http://localhost:8000/train/train_model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(config),
      })
        .then((response) => response.json())
        .then((job) => {
          // Expecting job schema: { id, status, status_details, error, created_at, expires_at }
          if (!job || !job.id) {
            console.error("Unexpected training job response:", job);
            alert("Training job created, but response format was unexpected. Check console.");
            return;
          }

          // Add training job to local state (newest first) and start polling
          // Add local timestamp for sorting if created_at is missing
          if (!job.created_at) {
            job._addedAt = Date.now();
          }
          trainJobs.unshift(job);
          renderJobs();
          startTrainJobPolling();
        })
        .catch((error) => {
          console.error("Failed to submit training configuration:", error);
          alert("Failed to submit training configuration. See console for details.");
        });
    });
  }

  // Initialize training config preview
  updateTrainingConfigPreview();
});


