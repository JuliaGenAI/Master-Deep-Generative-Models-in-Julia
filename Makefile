JL = julia

init:
	for case in 01_gpt2 02_Llama 03_deepseek 04_gemma models; do \
		echo "Instantiating $${case}"; \
		$(JL) --project=$${case} -e "using Pkg; Pkg.instantiate()"; \
	done

update:
	for case in 01_gpt2 02_Llama 03_deepseek 04_gemma models; do \
		echo "Updating $${case}"; \
		$(JL) --project=$${case} -e 'using Pkg; Pkg.update()'; \
	done

run:
	@echo "Running $${case}"; \
	$(JL) --project=$${case} -e "include(\"$${case}/main.jl\"); main()";

download:
	$(JL) --project=models -e "include(joinpath(@__DIR__, \"models\", \"download.jl\")); download_model(\"$${model}\"; overwrite=$${overwrite:-false})";

.PHONY: init update run download