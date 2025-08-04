#!/bin/bash
# Ground-Level Visual Localization System Launcher

set -e

echo "🌍 Ground-Level Visual Localization System"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${BLUE}📦 Installing dependencies...${NC}"
pip install -r requirements.txt > /dev/null 2>&1 || {
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
}

echo -e "${GREEN}✅ Dependencies installed successfully${NC}"

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

# Function to start API server
start_api() {
    echo -e "${BLUE}🚀 Starting API server...${NC}"
    
    if ! check_port 8000; then
        echo -e "${YELLOW}⚠️  Port 8000 is already in use${NC}"
        read -p "Kill existing process and restart? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            lsof -ti:8000 | xargs kill -9 2>/dev/null || true
            sleep 2
        else
            echo -e "${RED}❌ Cannot start API server${NC}"
            return 1
        fi
    fi
    
    cd src
    python -m api.main &
    API_PID=$!
    cd ..
    
    # Wait for API to start
    echo -e "${BLUE}⏳ Waiting for API to start...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ API server started successfully${NC}"
            echo -e "${GREEN}📖 API Documentation: http://localhost:8000/docs${NC}"
            return 0
        fi
        sleep 1
    done
    
    echo -e "${RED}❌ API server failed to start${NC}"
    return 1
}

# Function to start web interface
start_web() {
    echo -e "${BLUE}🌐 Starting web interface...${NC}"
    
    if ! check_port 8501; then
        echo -e "${YELLOW}⚠️  Port 8501 is already in use${NC}"
        read -p "Kill existing process and restart? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            lsof -ti:8501 | xargs kill -9 2>/dev/null || true
            sleep 2
        else
            echo -e "${RED}❌ Cannot start web interface${NC}"
            return 1
        fi
    fi
    
    cd web
    streamlit run app.py --server.headless true --server.port 8501 &
    WEB_PID=$!
    cd ..
    
    # Wait for web app to start
    echo -e "${BLUE}⏳ Waiting for web interface to start...${NC}"
    for i in {1..20}; do
        if curl -s http://localhost:8501 > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Web interface started successfully${NC}"
            echo -e "${GREEN}🌐 Web Interface: http://localhost:8501${NC}"
            return 0
        fi
        sleep 1
    done
    
    echo -e "${RED}❌ Web interface failed to start${NC}"
    return 1
}

# Function to check system status
check_status() {
    echo -e "${BLUE}🔍 Checking system status...${NC}"
    
    # Check API
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API Server: Running (http://localhost:8000)${NC}"
    else
        echo -e "${RED}❌ API Server: Not running${NC}"
    fi
    
    # Check Web Interface
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Web Interface: Running (http://localhost:8501)${NC}"
    else
        echo -e "${RED}❌ Web Interface: Not running${NC}"
    fi
    
    # Check database
    if [ -f "data/streetview.db" ]; then
        echo -e "${GREEN}✅ Database: Found${NC}"
    else
        echo -e "${YELLOW}⚠️  Database: Not found (system will create it on first use)${NC}"
    fi
}

# Function to stop all services
stop_services() {
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    
    # Kill processes on known ports
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:8501 | xargs kill -9 2>/dev/null || true
    
    # Kill background processes if we have their PIDs
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$WEB_PID" ]; then
        kill $WEB_PID 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ All services stopped${NC}"
}

# Function to setup database
setup_database() {
    echo -e "${BLUE}🗃️  Setting up database...${NC}"
    
    # Create data directories
    mkdir -p data/streetview data/embeddings data/cache
    
    echo -e "${GREEN}✅ Database directories created${NC}"
    echo -e "${YELLOW}📝 To populate database with street-view images:${NC}"
    echo -e "${YELLOW}   1. Use Google Street View API or Mapillary${NC}"
    echo -e "${YELLOW}   2. Call /database/rebuild endpoint${NC}"
    echo -e "${YELLOW}   3. Or use the DatabaseBuilder class directly${NC}"
}

# Function to run example
run_example() {
    echo -e "${BLUE}🧪 Running example localization...${NC}"
    
    # Check if API is running
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${RED}❌ API server is not running. Start it first with: $0 --api${NC}"
        return 1
    fi
    
    # Create example image if it doesn't exist
    if [ ! -f "example_image.jpg" ]; then
        echo -e "${YELLOW}📷 No example image found. Please add 'example_image.jpg' to test.${NC}"
        echo -e "${YELLOW}📝 You can download a street-view image and save it as 'example_image.jpg'${NC}"
        return 1
    fi
    
    echo -e "${BLUE}📤 Uploading example image for localization...${NC}"
    
    curl -X POST "http://localhost:8000/localize" \
         -F "file=@example_image.jpg" \
         -F "method=ensemble" \
         -F "include_debug=true" \
         | python3 -m json.tool
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --all          Start both API server and web interface"
    echo "  --api          Start only the API server"
    echo "  --web          Start only the web interface"
    echo "  --status       Check status of all services"
    echo "  --stop         Stop all running services"
    echo "  --setup        Setup database directories"
    echo "  --example      Run example localization (requires API running)"
    echo "  --install      Install dependencies only"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all       # Start complete system"
    echo "  $0 --api       # Start API only"
    echo "  $0 --status    # Check what's running"
    echo "  $0 --stop      # Stop everything"
    echo ""
    echo "After starting:"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  Web Interface: http://localhost:8501"
}

# Trap to cleanup on exit
trap stop_services EXIT

# Parse command line arguments
case "${1:-}" in
    --all)
        setup_database
        start_api
        sleep 3
        start_web
        echo ""
        echo -e "${GREEN}🎉 System started successfully!${NC}"
        echo -e "${GREEN}📖 API: http://localhost:8000/docs${NC}"
        echo -e "${GREEN}🌐 Web: http://localhost:8501${NC}"
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
        wait
        ;;
    --api)
        setup_database
        start_api
        echo ""
        echo -e "${GREEN}🎉 API server started!${NC}"
        echo -e "${GREEN}📖 Documentation: http://localhost:8000/docs${NC}"
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        wait
        ;;
    --web)
        start_web
        echo ""
        echo -e "${GREEN}🎉 Web interface started!${NC}"
        echo -e "${GREEN}🌐 Interface: http://localhost:8501${NC}"
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        wait
        ;;
    --status)
        check_status
        ;;
    --stop)
        stop_services
        exit 0
        ;;
    --setup)
        setup_database
        ;;
    --example)
        run_example
        ;;
    --install)
        echo -e "${GREEN}✅ Dependencies already installed${NC}"
        ;;
    --help)
        show_help
        ;;
    "")
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        show_help
        exit 1
        ;;
esac