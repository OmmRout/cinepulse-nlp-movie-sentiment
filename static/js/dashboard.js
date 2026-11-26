/**
 * CinePulse NLP Dashboard Client
 * Safe DOM rendering (no untrusted innerHTML), CSRF-protected API calls, Chart.js analytics.
 */
(function () {
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute("content") || "";

  const SAMPLE_REVIEWS = {
    positive:
      "An astonishing triumph of modern cinema. Every frame feels painterly and deliberate, while the lead actress delivers a heartbreaking, nuanced performance. The haunting orchestral score elevates the third act into something truly unforgettable.",
    negative:
      "What an exhausting, hollow mess. Despite an expensive visual budget, the screenplay is painfully predictable, the pacing drags endlessly through pointless subplots, and the wooden dialogue ruins every dramatic moment. A complete waste of two hours.",
    negation:
      "At first I expected another shallow blockbuster, and the opening twenty minutes were not very promising. However, the film never falls into cheap clichés—it is not boring at all, and the director refuses to take easy shortcuts.",
    mixed:
      "The cinematography and sound design are gorgeous, and the cast clearly gave strong performances. Unfortunately, the messy final act and rushed ending undermine an otherwise compelling mystery, leaving it intriguing but flawed.",
  };

  let overviewData = null;
  let lastBatchRows = [];
  const charts = {};

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function clearNode(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function destroyChart(id) {
    if (charts[id]) {
      charts[id].destroy();
      delete charts[id];
    }
  }

  // Tab navigation
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const target = document.getElementById("tab-" + btn.dataset.tab);
      if (target) target.classList.add("active");
    });
  });

  // Live Review Composer helpers
  const reviewInput = document.getElementById("review-input");
  const charCounter = document.getElementById("char-counter");
  const singleError = document.getElementById("single-error-banner");

  function updateCounter() {
    charCounter.textContent = `${reviewInput.value.length.toLocaleString()} / 10,000 chars`;
  }
  reviewInput.addEventListener("input", updateCounter);

  document.querySelectorAll("[data-sample]").forEach((btn) => {
    btn.addEventListener("click", () => {
      reviewInput.value = SAMPLE_REVIEWS[btn.dataset.sample] || "";
      updateCounter();
      singleError.classList.add("hidden");
    });
  });

  document.getElementById("clear-review-btn").addEventListener("click", () => {
    reviewInput.value = "";
    updateCounter();
    singleError.classList.add("hidden");
  });

  // Single Review Analysis Submission
  document.getElementById("single-review-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    singleError.classList.add("hidden");
    const text = reviewInput.value.trim();
    if (!text) {
      singleError.textContent = "Please enter a movie review before analyzing.";
      singleError.classList.remove("hidden");
      return;
    }

    const analyzeBtn = document.getElementById("analyze-btn");
    analyzeBtn.disabled = true;
    analyzeBtn.querySelector(".btn-label").textContent = "Analyzing...";

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRF-Token": csrfToken,
        },
        body: JSON.stringify({
          review_text: text,
          model: document.getElementById("single-model-select").value,
        }),
      });
      const isJson = (resp.headers.get("content-type") || "").includes("application/json");
      const data = isJson ? await resp.json() : null;
      if (!resp.ok || !data) {
        throw new Error((data && data.error) || `Server returned HTTP ${resp.status}. Please retry.`);
      }
      renderSinglePrediction(data);
    } catch (err) {
      singleError.textContent = err.message;
      singleError.classList.remove("hidden");
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.querySelector(".btn-label").textContent = "Analyze Sentiment";
    }
  });

  function renderSinglePrediction(data) {
    document.getElementById("verdict-placeholder").classList.add("hidden");
    document.getElementById("verdict-results").classList.remove("hidden");
    document.getElementById("single-deep-dive").classList.remove("hidden");

    document.getElementById("res-model-label").textContent = data.active_model;
    const badge = document.getElementById("res-verdict-badge");
    badge.textContent = data.verdict;
    badge.className = "verdict-badge " + (data.verdict === "Positive" ? "pos" : "neg");

    document.getElementById("res-intensity-tag").textContent = data.intensity;
    document.getElementById("res-star-rating").textContent = data.estimated_rating.toFixed(1);
    document.getElementById("res-prob-pos").textContent = (data.prob_positive * 100).toFixed(1) + "%";
    document.getElementById("res-prob-neg").textContent = (data.prob_negative * 100).toFixed(1) + "%";
    document.getElementById("res-confidence").textContent = (data.confidence * 100).toFixed(1) + "%";
    document.getElementById("res-word-count").textContent = data.word_count;

    const bar = document.getElementById("res-polarity-bar");
    bar.style.width = `${Math.round(data.prob_positive * 100)}%`;

    // Model breakdown bars
    const modelBars = document.getElementById("res-model-bars");
    clearNode(modelBars);
    data.model_comparison.forEach((m) => {
      const row = el("div", "model-bar-row");
      row.appendChild(el("span", "", m.name));
      if (m.unavailable) {
        row.appendChild(el("span", "muted", "Unavailable"));
        row.appendChild(el("span", "mono-val", "—"));
      } else {
        const track = el("div", "bar-track");
        const fill = el("div", m.prediction === "Positive" ? "bar-fill-pos" : "bar-fill-neg");
        fill.style.width = `${Math.round(m.prob_positive * 100)}%`;
        track.appendChild(fill);
        row.appendChild(track);
        row.appendChild(
          el("span", "mono-val", `${m.prediction.slice(0, 3)} ${(m.prob_positive * 100).toFixed(0)}% · ${m.latency_ms}ms`)
        );
      }
      modelBars.appendChild(row);
    });

    // Token attributions
    const tokContainer = document.getElementById("res-token-attributions");
    clearNode(tokContainer);
    if (!data.token_attributions.length) {
      tokContainer.appendChild(el("p", "muted", "No overlapping vocabulary tokens in TF-IDF lexicon."));
    } else {
      data.token_attributions.forEach((t) => {
        const item = el("div", `token-pill-row ${t.direction === "positive" ? "pos" : "neg"}`);
        item.appendChild(el("span", "", t.token));
        const sign = t.contribution >= 0 ? "+" : "";
        item.appendChild(el("span", "mono-val", `${sign}${t.contribution.toFixed(4)}`));
        tokContainer.appendChild(item);
      });
    }

    // Movie Aspects
    const aspectContainer = document.getElementById("res-aspects");
    clearNode(aspectContainer);
    if (!data.aspects.length) {
      aspectContainer.appendChild(el("p", "muted", "No specific movie aspects (acting, plot, pacing, visuals, audio) explicitly named."));
    } else {
      data.aspects.forEach((a) => {
        const card = el("div", "aspect-card");
        const head = el("div", "aspect-head");
        head.appendChild(el("span", "", a.aspect));
        const toneCls = a.tone === "Positive" ? "pos" : a.tone === "Negative" ? "neg" : "mix";
        head.appendChild(el("span", `aspect-tone ${toneCls}`, a.tone));
        card.appendChild(head);
        card.appendChild(el("div", "aspect-kw", `Matched terms: ${a.matched_keywords.join(", ")} (${a.mentions} sentence${a.mentions > 1 ? "s" : ""})`));
        aspectContainer.appendChild(card);
      });
    }

    // Preprocessing Audit Trace
    const traceBox = document.getElementById("res-preprocessing");
    clearNode(traceBox);
    const steps = [
      ["Normalized Text", data.preprocessing.normalized_preview || "(empty)"],
      [
        "Token Pipeline",
        `${data.preprocessing.token_count_raw} raw tokens -> ${data.preprocessing.token_count_after_stopwords} after negation-aware stopword filter`,
      ],
      ["Lemmatized Tokens", data.preprocessing.lemmatized_sample.join(" · ") || "(none)"],
    ];
    steps.forEach(([label, val]) => {
      const box = el("div", "trace-item");
      box.appendChild(el("strong", "", label));
      box.appendChild(el("code", "", val));
      traceBox.appendChild(box);
    });
  }

  // Batch Upload Studio
  const fileInput = document.getElementById("batch-file-input");
  const dropzone = document.getElementById("batch-dropzone");
  const fileNameLabel = document.getElementById("batch-file-name");
  const batchError = document.getElementById("batch-error-banner");

  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      fileNameLabel.textContent = `Selected: ${fileInput.files[0].name} (${(fileInput.files[0].size / 1024).toFixed(1)} KB)`;
    }
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      fileNameLabel.textContent = `Selected: ${fileInput.files[0].name}`;
    }
  });

  async function submitBatchFile(fileObj) {
    batchError.classList.add("hidden");
    const submitBtn = document.getElementById("batch-submit-btn");
    submitBtn.disabled = true;
    submitBtn.textContent = "Scoring Batch...";

    const formData = new FormData();
    formData.append("file", fileObj);
    formData.append("model", document.getElementById("batch-model-select").value);

    try {
      const resp = await fetch("/api/batch", {
        method: "POST",
        headers: { "X-CSRF-Token": csrfToken },
        body: formData,
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Batch processing failed.");
      renderBatchResults(data);
    } catch (err) {
      batchError.textContent = err.message;
      batchError.classList.remove("hidden");
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Run Batch Screening";
    }
  }

  document.getElementById("batch-upload-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!fileInput.files.length) {
      batchError.textContent = "Please choose a .csv or .txt review file first.";
      batchError.classList.remove("hidden");
      return;
    }
    submitBatchFile(fileInput.files[0]);
  });

  document.getElementById("download-sample-batch-btn").addEventListener("click", () => {
    const sampleCsv = [
      "review",
      "\"Brilliant pacing and haunting cinematography made this thriller unforgettable.\"",
      "\"Boring dialogue and terrible acting ruined an otherwise decent premise.\"",
      "\"Not bad at all; the soundtrack carried the emotional weight beautifully.\"",
      "\"A predictable sequel with flat characters and awful CGI effects.\"",
      "\"The director crafted a taut, deeply moving story with Oscar-worthy performances.\"",
      "\"Slow, pretentious, and way too long for what little plot it had.\"",
      "\"Although the ending felt slightly rushed, the acting and screenplay were superb.\"",
      "\"One of the worst comedies I have ever endured in a theater.\"",
    ].join("\n");
    const blob = new Blob([sampleCsv], { type: "text/csv" });
    const sampleFile = new File([blob], "sample_movie_reviews.csv", { type: "text/csv" });
    fileNameLabel.textContent = "Running built-in sample: sample_movie_reviews.csv";
    submitBatchFile(sampleFile);
  });

  function renderBatchResults(data) {
    lastBatchRows = data.rows || [];
    document.getElementById("batch-summary-subtitle").textContent = `Scored with ${data.model_used}`;
    document.getElementById("b-total").textContent = data.total_reviews;
    document.getElementById("b-pos-pct").textContent = `${data.positive_pct}% (${data.positive_count}/${data.total_reviews})`;
    document.getElementById("b-avg-rating").textContent = `${data.avg_estimated_rating} / 10`;
    document.getElementById("b-avg-conf").textContent = `${(data.avg_confidence * 100).toFixed(1)}%`;
    document.getElementById("export-batch-btn").classList.remove("hidden");
    document.getElementById("batch-results-card").classList.remove("hidden");

    // Rating Histogram Chart
    destroyChart("batchRating");
    const labels = Object.keys(data.rating_histogram);
    const values = Object.values(data.rating_histogram);
    charts.batchRating = new Chart(document.getElementById("batch-rating-chart"), {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "Reviews in Estimated Rating Band",
            data: values,
            backgroundColor: labels.map((_, idx) => (idx < 4 ? "rgba(223, 94, 82, 0.78)" : "rgba(74, 195, 150, 0.78)")),
            borderRadius: 6,
          },
        ],
      },
      options: chartBaseOptions("Estimated IMDb Rating Band", "Review Count"),
    });

    const tbody = document.getElementById("batch-table-body");
    clearNode(tbody);
    lastBatchRows.forEach((row) => {
      const tr = el("tr");
      tr.appendChild(el("td", "mono-val", row.index));
      const tdSent = el("td");
      tdSent.appendChild(el("span", `tag-sent ${row.sentiment === "Positive" ? "pos" : "neg"}`, row.sentiment));
      tr.appendChild(tdSent);
      tr.appendChild(el("td", "mono-val", `${(row.prob_positive * 100).toFixed(1)}%`));
      tr.appendChild(el("td", "mono-val", row.estimated_rating.toFixed(1)));
      tr.appendChild(el("td", "", row.aspects.length ? row.aspects.join(", ") : "General"));
      tr.appendChild(el("td", "", row.excerpt));
      tbody.appendChild(tr);
    });
  }

  document.getElementById("export-batch-btn").addEventListener("click", async () => {
    if (!lastBatchRows.length) return;
    const resp = await fetch("/api/export", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrfToken,
      },
      body: JSON.stringify({ rows: lastBatchRows }),
    });
    if (!resp.ok) return;
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "cinepulse_sentiment_results.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  });

  // Chart.js styling aligned with Warm Obsidian & Gold palette
  function chartBaseOptions(xLabel, yLabel) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#e8dacd", font: { family: "Plus Jakarta Sans", size: 11 } } },
      },
      scales: {
        x: {
          title: { display: Boolean(xLabel), text: xLabel, color: "#c5b4a7" },
          ticks: { color: "#c5b4a7", font: { size: 11 } },
          grid: { color: "rgba(229, 178, 93, 0.08)" },
        },
        y: {
          title: { display: Boolean(yLabel), text: yLabel, color: "#c5b4a7" },
          ticks: { color: "#c5b4a7", font: { size: 11 } },
          grid: { color: "rgba(229, 178, 93, 0.08)" },
        },
      },
    };
  }

  async function loadOverview() {
    try {
      const [overviewResult, statusResult] = await Promise.allSettled([
        fetch("/api/overview"),
        fetch("/api/status"),
      ]);

      const badge = document.getElementById("badge-models");
      if (statusResult.status === "fulfilled" && statusResult.value.ok) {
        const health = await statusResult.value.json();
        badge.textContent = health.transformer_loaded
          ? "DistilBERT ONNX + 5 Classical Ready"
          : "Classical ML Ready (Transformer Lazy)";
      } else {
        badge.textContent = "5 Classical + DistilBERT Ready";
      }

      if (overviewResult.status === "fulfilled" && overviewResult.value.ok) {
        overviewData = await overviewResult.value.json();
        renderOverview(overviewData);
      }
    } catch (err) {
      console.error("Failed loading overview:", err);
    }
  }

  function renderOverview(data) {
    const ds = data.dataset || {};
    const ev = data.evaluation || {};
    const ea = data.error_analysis || {};
    const proj = data.embedding_projection || {};
    const asp = data.aspects || {};

    if (ds.cleaned_records) {
      document.getElementById("stat-corpus-size").textContent = ds.cleaned_records.toLocaleString();
      document.getElementById("stat-corpus-source").textContent = ds.dataset_name || "Stanford IMDb";
      document.getElementById("stat-avg-words").textContent = ds.word_count_stats?.mean ?? "—";
    }
    if (ev.test_set_size) {
      document.getElementById("stat-test-size").textContent = ev.test_set_size.toLocaleString();
      document.getElementById("stat-best-model").textContent = ev.best_model_name || "—";
      const best = (ev.models || []).find((m) => m.key === ev.best_model_key);
      if (best) {
        document.getElementById("stat-best-f1").textContent = `F1 ${(best.metrics.f1_score * 100).toFixed(2)}% · AUC ${best.metrics.roc_auc.toFixed(3)}`;
      }
    }

    // Benchmark Table
    const tbody = document.getElementById("benchmark-table-body");
    clearNode(tbody);
    const models = ev.models || [];
    models.forEach((m) => {
      const tr = el("tr", m.key === ev.best_model_key ? "best-row" : "");
      tr.appendChild(el("td", "", m.name));
      tr.appendChild(el("td", "", m.family));
      tr.appendChild(el("td", "", m.representation));
      tr.appendChild(el("td", "mono-val", (m.metrics.accuracy * 100).toFixed(2) + "%"));
      tr.appendChild(el("td", "mono-val", (m.metrics.precision * 100).toFixed(2) + "%"));
      tr.appendChild(el("td", "mono-val", (m.metrics.recall * 100).toFixed(2) + "%"));
      tr.appendChild(el("td", "mono-val", (m.metrics.f1_score * 100).toFixed(2) + "%"));
      tr.appendChild(el("td", "mono-val", m.metrics.roc_auc.toFixed(4)));
      tr.appendChild(el("td", "mono-val", m.metrics.latency_ms_per_review.toFixed(2)));
      tbody.appendChild(tr);
    });

    // Model Comparison Chart
    if (models.length) {
      destroyChart("modelComp");
      charts.modelComp = new Chart(document.getElementById("model-comparison-chart"), {
        type: "bar",
        data: {
          labels: models.map((m) => m.name),
          datasets: [
            {
              label: "Accuracy (%)",
              data: models.map((m) => +(m.metrics.accuracy * 100).toFixed(2)),
              backgroundColor: "rgba(229, 178, 93, 0.85)",
              borderRadius: 6,
            },
            {
              label: "F1-Score (%)",
              data: models.map((m) => +(m.metrics.f1_score * 100).toFixed(2)),
              backgroundColor: "rgba(74, 195, 150, 0.82)",
              borderRadius: 6,
            },
          ],
        },
        options: {
          ...chartBaseOptions("", "Score (%)"),
          scales: {
            ...chartBaseOptions("", "Score (%)").scales,
            y: { min: 60, max: 100, ticks: { color: "#c5b4a7" }, grid: { color: "rgba(229, 178, 93, 0.08)" } },
          },
        },
      });

      // ROC Curve Chart
      const palette = ["#df5e52", "#e08e52", "#e5b25d", "#f6d38a", "#a876b8", "#4ac396"];
      destroyChart("rocCurve");
      charts.rocCurve = new Chart(document.getElementById("roc-curve-chart"), {
        type: "line",
        data: {
          datasets: models.map((m, idx) => ({
            label: `${m.name} (${m.metrics.roc_auc.toFixed(3)})`,
            data: m.metrics.roc_curve.fpr.map((x, i) => ({ x, y: m.metrics.roc_curve.tpr[i] })),
            borderColor: palette[idx % palette.length],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.25,
          })),
        },
        options: {
          ...chartBaseOptions("False Positive Rate", "True Positive Rate"),
          scales: {
            x: { type: "linear", min: 0, max: 1, ticks: { color: "#c5b4a7" }, grid: { color: "rgba(229, 178, 93, 0.08)" } },
            y: { type: "linear", min: 0, max: 1, ticks: { color: "#c5b4a7" }, grid: { color: "rgba(229, 178, 93, 0.08)" } },
          },
        },
      });

      // CM selector
      const cmSelect = document.getElementById("cm-model-select");
      clearNode(cmSelect);
      models.forEach((m) => {
        const opt = el("option", "", m.name);
        opt.value = m.key;
        if (m.key === ev.best_model_key) opt.selected = true;
        cmSelect.appendChild(opt);
      });
      cmSelect.addEventListener("change", () => renderConfusionMatrix(models, cmSelect.value));
      renderConfusionMatrix(models, cmSelect.value);
    }

    // 2D Embedding Scatter Plot
    if (proj.points && proj.points.length) {
      const posPts = proj.points.filter((p) => p.sentiment === "positive").map((p) => ({ x: p.x, y: p.y }));
      const negPts = proj.points.filter((p) => p.sentiment === "negative").map((p) => ({ x: p.x, y: p.y }));
      destroyChart("scatter");
      charts.scatter = new Chart(document.getElementById("embedding-scatter-chart"), {
        type: "scatter",
        data: {
          datasets: [
            { label: "Positive Reviews", data: posPts, backgroundColor: "rgba(74, 195, 150, 0.65)", pointRadius: 3.5 },
            { label: "Negative Reviews", data: negPts, backgroundColor: "rgba(223, 94, 82, 0.65)", pointRadius: 3.5 },
          ],
        },
        options: chartBaseOptions("PCA Component 1", "PCA Component 2"),
      });
    }

    // Top Predictive N-grams
    const ngContainer = document.getElementById("top-ngrams-container");
    clearNode(ngContainer);
    if (ev.top_ngrams) {
      const posCol = el("div", "ngram-col");
      posCol.appendChild(el("h4", "", "Positive Signals"));
      ev.top_ngrams.positive.slice(0, 10).forEach((item) => {
        const row = el("div", "token-pill-row pos");
        row.appendChild(el("span", "", item.term));
        row.appendChild(el("span", "mono-val", "+" + item.weight.toFixed(2)));
        posCol.appendChild(row);
      });

      const negCol = el("div", "ngram-col");
      negCol.appendChild(el("h4", "", "Negative Signals"));
      ev.top_ngrams.negative.slice(0, 10).forEach((item) => {
        const row = el("div", "token-pill-row neg");
        row.appendChild(el("span", "", item.term));
        row.appendChild(el("span", "mono-val", item.weight.toFixed(2)));
        negCol.appendChild(row);
      });

      ngContainer.appendChild(posCol);
      ngContainer.appendChild(negCol);
    }

    // Corpus Aspects
    if (asp.aspects && asp.aspects.length) {
      destroyChart("aspectCorpus");
      charts.aspectCorpus = new Chart(document.getElementById("aspect-corpus-chart"), {
        type: "bar",
        data: {
          labels: asp.aspects.map((a) => a.aspect),
          datasets: [
            {
              label: "Mention Share (%)",
              data: asp.aspects.map((a) => a.share_pct),
              backgroundColor: "rgba(229, 178, 93, 0.8)",
              borderRadius: 6,
            },
            {
              label: "Positive Rate (%)",
              data: asp.aspects.map((a) => +(a.positive_rate * 100).toFixed(1)),
              backgroundColor: "rgba(74, 195, 150, 0.8)",
              borderRadius: 6,
            },
          ],
        },
        options: chartBaseOptions("", "Percentage (%)"),
      });
    }

    // Error Analysis
    renderErrorAnalysis(ea);

    // Dataset Sample Records
    const sampList = document.getElementById("dataset-samples-list");
    clearNode(sampList);
    (ds.sample_records || []).slice(0, 4).forEach((rec) => {
      const card = el("div", "trace-item");
      card.appendChild(
        el("strong", "", `${rec.review_id} · ${rec.sentiment.toUpperCase()} · Raw Rating ${rec.star_rating_raw}/10 · ${rec.word_count} words`)
      );
      card.appendChild(el("code", "", rec.excerpt));
      sampList.appendChild(card);
    });
  }

  function renderConfusionMatrix(models, modelKey) {
    const model = models.find((m) => m.key === modelKey) || models[0];
    if (!model) return;
    const [[tn, fp], [fn, tp]] = model.metrics.confusion_matrix;
    const grid = document.getElementById("cm-grid-container");
    clearNode(grid);

    [
      ["True Negative (TN)", tn, "hit"],
      ["False Positive (FP)", fp, "miss"],
      ["False Negative (FN)", fn, "miss"],
      ["True Positive (TP)", tp, "hit"],
    ].forEach(([label, count, cls]) => {
      const cell = el("div", `cm-cell ${cls}`);
      cell.appendChild(el("span", "", label));
      cell.appendChild(el("strong", "", count));
      grid.appendChild(cell);
    });

    const sd = model.metrics.score_distribution;
    destroyChart("scoreDist");
    charts.scoreDist = new Chart(document.getElementById("score-dist-chart"), {
      type: "bar",
      data: {
        labels: sd.bins,
        datasets: [
          { label: "Actual Negative", data: sd.true_negative_counts, backgroundColor: "rgba(223, 94, 82, 0.72)" },
          { label: "Actual Positive", data: sd.true_positive_counts, backgroundColor: "rgba(74, 195, 150, 0.72)" },
        ],
      },
      options: chartBaseOptions("Predicted P(Positive) Bin", "Test Reviews"),
    });
  }

  function renderErrorAnalysis(ea) {
    const summaryBox = document.getElementById("error-summary-stats");
    clearNode(summaryBox);
    if (ea.total_test_samples) {
      const items = [
        ["Test Reviews Audited", `${ea.total_test_samples} unseen IMDb reviews`],
        ["TF-IDF + LogReg Misclassifications", `${ea.logreg_error_count} errors`],
        [
          "DistilBERT ONNX Misclassifications",
          ea.transformer_error_count !== null ? `${ea.transformer_error_count} errors` : "Not evaluated",
        ],
        [
          "Key Diagnostic Takeaway",
          "Errors cluster around concessive 'but/however' reviews, long plot summaries, and sarcastic praise — contexts where bidirectional transformer attention outperforms linear n-grams.",
        ],
      ];
      items.forEach(([k, v]) => {
        const box = el("div", "trace-item");
        box.appendChild(el("strong", "", k));
        box.appendChild(el("span", "muted", v));
        summaryBox.appendChild(box);
      });
    }

    if (ea.cause_breakdown) {
      destroyChart("errorCause");
      charts.errorCause = new Chart(document.getElementById("error-cause-chart"), {
        type: "doughnut",
        data: {
          labels: Object.keys(ea.cause_breakdown),
          datasets: [
            {
              data: Object.values(ea.cause_breakdown),
              backgroundColor: ["#e5b25d", "#df5e52", "#a876b8", "#4ac396", "#e08e52", "#8f7e74"],
              borderColor: "#140f11",
              borderWidth: 2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "right", labels: { color: "#e8dacd", font: { size: 11 } } },
          },
        },
      });
    }

    const filterButtons = document.querySelectorAll("[data-errtab]");
    filterButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        filterButtons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        renderErrorCases(ea[btn.dataset.errtab] || []);
      });
    });
    renderErrorCases(ea.false_positives || []);
  }

  function renderErrorCases(cases) {
    const grid = document.getElementById("error-cases-grid");
    clearNode(grid);
    if (!cases.length) {
      grid.appendChild(el("p", "muted", "No cases recorded in this category."));
      return;
    }
    cases.forEach((c) => {
      const card = el("div", "error-case-card");
      const meta = el("div", "error-case-meta");
      meta.appendChild(el("span", "mono-val", `True: ${c.true_sentiment} (${c.star_rating_raw}/10) · LogReg: ${c.logreg_pred} (${(c.logreg_prob_pos * 100).toFixed(0)}%)`));
      if (c.transformer_pred) {
        meta.appendChild(el("span", "mono-val", `DistilBERT: ${c.transformer_pred} (${(c.transformer_prob_pos * 100).toFixed(0)}%)`));
      }
      card.appendChild(meta);
      card.appendChild(el("span", "cause-badge", c.probable_cause));
      card.appendChild(el("p", "muted", c.excerpt));
      card.appendChild(el("small", "", `Diagnostic: ${c.explanation}`));
      grid.appendChild(card);
    });
  }

  loadOverview();
})();
