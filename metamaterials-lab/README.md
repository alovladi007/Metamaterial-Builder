# Metamaterials Research Lab

## Quick Start

### Docker Setup
```bash
cd docker
docker-compose up
```

### Manual Setup

#### Backend:
```bash
cd api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

#### Frontend:
```bash
cd client
npm install
npm run dev
```

### Access

- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs