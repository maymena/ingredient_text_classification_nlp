#!/bin/bash

# Check if indexing has been done
if [ ! -f /app/.indexed ]; then
    echo "Running initial indexing..."
    python web/index_data.py --opensearch_url "$OPENSEARCH_URL"
    # Create a marker file to indicate indexing is done
    touch /app/.indexed
    echo "Indexing completed"
else
    echo "Indexing already completed, skipping..."
fi

# Start the Flask application
python web/app.py 