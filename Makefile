# ============================================================================
# DML Benchmark — Makefile
# ============================================================================
# Quick reference:
#   make install     Install package in editable mode
#   make test        Run full test suite
#   make smoke       Run smoke test (<2 min)
#   make analysis    Run full analysis (all 19 sections)
#   make figures     Generate all figures (PDF)
#   make zip         Create timestamped backup zip
#   make clean       Remove generated files
# ============================================================================

.PHONY: install test smoke analysis figures zip submission-zip clean help

PYTHON ?= python3
TIERS ?= 1 2 3 4

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

test:  ## Run full test suite (158 tests)
	$(PYTHON) -m pytest tests/ -v --tb=short

smoke:  ## Run smoke test — quick sanity check (<2 min)
	$(PYTHON) run_smoke_test.py

analysis:  ## Run full analysis (all 19 sections, tiers 1-4)
	$(PYTHON) analyze_results_v2.py --tiers $(TIERS) --section all

figures:  ## Generate all figures (PDF format)
	$(PYTHON) create_figures_v2.py --tiers $(TIERS) --figure all --format pdf

zip:  ## Create timestamped backup zip (excludes repos, pycache, history, data, drafts)
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S) && \
	ZIP_NAME="../o_df_ml_backup_$${TIMESTAMP}.zip" && \
	cd .. && zip -r "$$ZIP_NAME" o_df_ml/ \
		-x "o_df_ml/repos/*" \
		   "o_df_ml/__pycache__/*" "o_df_ml/dml_benchmark/__pycache__/*" \
		   "o_df_ml/.git/*" "o_df_ml/.git/lfs/*" \
		   "o_df_ml/data/*" \
		   "o_df_ml/logs/*" \
		   "o_df_ml/papers/*" \
		   "o_df_ml/*.zip" "o_df_ml/*.tar.gz" && \
	echo "Backup saved to $$ZIP_NAME"

submission-zip:  ## Supplementary tarball for E&D submission (anonymous, ≤100 MB cap)
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S) && \
	ZIP_NAME="../o_df_ml_supplementary_$${TIMESTAMP}.zip" && \
	cd .. && zip -r "$$ZIP_NAME" o_df_ml/ \
		-x "o_df_ml/.git/*" "o_df_ml/.git/lfs/*" \
		   "o_df_ml/repos/*" \
		   "o_df_ml/data/*" "o_df_ml/logs/*" "o_df_ml/papers/*" \
		   "o_df_ml/__pycache__/*" "o_df_ml/dml_benchmark/__pycache__/*" \
		   "o_df_ml/*.zip" "o_df_ml/*.tar.gz" \
		   "o_df_ml/_zip_extract/*" "o_df_ml/_audit/*" \
		   "o_df_ml/manual_export/*" "o_df_ml/o_df_ml_export*/*" \
		   "o_df_ml/paper_export_split/*" "o_df_ml/tmp_files/*" \
		   "o_df_ml/tmp_pipeline2_caches/*" "o_df_ml/retrosynth_tmp/*" \
		   "o_df_ml/notebooks/*" && \
	ls -lh "$$ZIP_NAME" && \
	echo "Supplementary tarball saved to $$ZIP_NAME" && \
	echo "If file size > 100 MB you MUST drop more — track cap is 100 MB."

clean:  ## Remove generated files (figures, pycache)
	rm -rf figures/*.pdf figures/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache build dist *.egg-info
