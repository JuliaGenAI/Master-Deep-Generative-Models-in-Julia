using Downloads
using JSON3

"""Discover what files are available in a HuggingFace model repository"""
function discover_model_files(model_name; hf_token)
    # Set up headers for authentication if token is available
    headers = Dict{String, String}()
    if hf_token !== nothing
        headers["Authorization"] = "Bearer $hf_token"
    end
    
    try
        # Use the HuggingFace API to list files
        api_url = "https://huggingface.co/api/models/$model_name/tree/main"
        
        # Download file listing
        response_io = IOBuffer()
        if !isempty(headers)
            Downloads.download(api_url, response_io, headers=headers)
        else
            Downloads.download(api_url, response_io)
        end
        
        # Parse the JSON response
        response_str = String(take!(response_io))
        files_data = JSON3.read(response_str)
        
        # Extract file names
        model_files = []
        tokenizer_file = nothing
        config_file = nothing
        
        for file_info in files_data
            filename = file_info.path
            
            if endswith(filename, ".safetensors")
                push!(model_files, filename)
            elseif filename == "tokenizer.json"
                tokenizer_file = filename
            elseif filename == "config.json"
                config_file = filename
            end
        end
        
        return (
            model_files = model_files,
            tokenizer_file = tokenizer_file,
            config_file = config_file,
            all_files = [f.path for f in files_data]
        )
        
    catch e
        println("⚠️  Could not discover files via API, falling back to standard files")
        println("   Error: $e")
        
        # Fallback to standard file names
        return (
            model_files = ["model.safetensors"],
            tokenizer_file = "tokenizer.json", 
            config_file = "config.json",
            all_files = String[]
        )
    end
end

function download_model(model_name; save_dir=joinpath(@__DIR__, model_name), overwrite=false, hf_token=get(ENV, "HF_TOKEN", nothing))
    println("Downloading model: $model_name")
    println("Save directory: $save_dir")
    
    # Create the directory if it doesn't exist
    if isdir(save_dir) && !overwrite
        println("Model already exists. Use `overwrite=true` to overwrite.")
        return save_dir
    end

    mkpath(save_dir)
    
    # Discover what files are available
    println("🔍 Discovering available model files...")
    discovered_files = discover_model_files(model_name; hf_token=hf_token)
    
    println("Found files:")
    println("  Config: $(discovered_files.config_file)")
    println("  Tokenizer: $(discovered_files.tokenizer_file)")
    println("  Model files: $(length(discovered_files.model_files)) SafeTensors file(s)")
    for file in discovered_files.model_files
        println("    - $file")
    end
    
    # Base URL for HuggingFace model files
    base_url = "https://huggingface.co/$model_name/resolve/main"
    
    # Set up headers for authentication if token is available
    headers = Dict{String, String}()
    if hf_token !== nothing
        headers["Authorization"] = "Bearer $hf_token"
        println("🔐 Using HuggingFace authentication token")
    else
        println("⚠️  No HuggingFace token found - trying without authentication")
        println("   If download fails, get a token from https://huggingface.co/settings/tokens")
        println("   and set it as: export HF_TOKEN='your_token_here'")
    end
    
    # Build list of files to download
    files_to_download = []
    
    # Add config and tokenizer files
    if discovered_files.config_file !== nothing
        push!(files_to_download, (discovered_files.config_file, discovered_files.config_file))
    end
    
    if discovered_files.tokenizer_file !== nothing
        push!(files_to_download, (discovered_files.tokenizer_file, discovered_files.tokenizer_file))
    end
    
    # Add all model files
    for model_file in discovered_files.model_files
        push!(files_to_download, (model_file, model_file))
    end
    
    try
        for (remote_file, local_file) in files_to_download
            local_path = joinpath(save_dir, local_file)
            remote_url = "$base_url/$remote_file"
            
            println("Downloading $remote_file...")
            
            # Download with or without authentication headers
            if !isempty(headers)
                Downloads.download(remote_url, local_path, headers=headers)
            else
                Downloads.download(remote_url, local_path)
            end
            
            println("✅ Saved: $local_path")
        end
        
        println("\n🎉 Model downloaded successfully!")
        println("Model files saved in: $save_dir")
        
        # Verify files exist
        println("\nVerifying downloaded files:")
        for (_, local_file) in files_to_download
            local_path = joinpath(save_dir, local_file)
            if isfile(local_path)
                size_mb = round(stat(local_path).size / 1024 / 1024, digits=2)
                println("✅ $local_file ($size_mb MB)")
            else
                println("❌ $local_file - Missing!")
            end
        end
        
        return save_dir
        
    catch e
        error_msg = string(e)
        if occursin("403", error_msg) || occursin("401", error_msg)
            println("❌ Authentication failed!")
            println("🔑 This model requires a HuggingFace access token.")
            println("   1. Go to https://huggingface.co/settings/tokens")
            println("   2. Create a token with 'Read' permissions")  
            println("   3. Accept the model's license agreement if required")
            println("   4. Set the token: export HF_TOKEN='your_token_here'")
            println("   5. Run the download again")
        else
            println("❌ Error downloading model: $e")
        end
        rethrow(e)
    end
end