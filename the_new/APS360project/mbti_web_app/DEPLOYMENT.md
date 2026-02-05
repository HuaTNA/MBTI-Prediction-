# Cloud Deployment Guide

## Overview

This guide shows you how to deploy the MBTI Assessment app to the cloud and manage the database.

## 🌐 Deployment Options

### Option 1: Heroku (Easiest)

**Pros**: Free tier, easy deployment, includes PostgreSQL
**Cons**: Limited free hours, app sleeps after 30min inactivity

**Steps**:

1. **Install Heroku CLI**
   ```bash
   # Download from: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login and Create App**
   ```bash
   heroku login
   heroku create your-mbti-app
   ```

3. **Add PostgreSQL**
   ```bash
   heroku addons:create heroku-postgresql:mini
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **Initialize Database**
   ```bash
   heroku run python setup_database.py
   ```

### Option 2: Render (Recommended for students)

**Pros**: Free tier, always-on, modern platform
**Cons**: Slower cold starts

**Steps**:

1. Go to https://render.com and sign up
2. Create new **Web Service** from your GitHub repo
3. Create new **PostgreSQL** database (free tier)
4. Set environment variable:
   ```
   DATABASE_URL = <your-render-postgres-url>
   ```
5. Deploy!

### Option 3: Railway

**Pros**: Simple, good free tier, includes PostgreSQL
**Cons**: Requires credit card for free tier

Similar steps to Render.

## 📊 Database Management for Cloud Deployment

### Strategy A: Start Fresh (Recommended for new apps)

1. Deploy app with empty database
2. Let users create new data
3. Old local data stays local

```bash
# On cloud server
python setup_database.py  # Creates empty tables
```

### Strategy B: Migrate Existing Data

1. **Export local database**
   ```bash
   python backup_database.py
   ```

2. **Upload backup to cloud** (via secure file transfer)

3. **Restore on cloud**
   ```bash
   # On cloud server
   python restore_database.py database_backup_20260205.json
   ```

### Strategy C: Dual Database Setup

Keep separate databases for:
- **Development** (local): Your testing data
- **Production** (cloud): Real user data

```bash
# .env.development (local)
DATABASE_URL=postgresql://postgres:password@localhost:5432/mbti_predictions

# .env.production (cloud)
DATABASE_URL=postgresql://user:pass@cloud-host:5432/mbti_predictions
```

## 👥 Team Collaboration with Database

### Scenario 1: Shared Cloud Database (Best)

Both you and your friend:
1. Use the same cloud database URL
2. Both can see all data
3. No sync needed

```bash
# Both .env files
DATABASE_URL=postgresql://user:pass@neon.tech/mbti_predictions
```

### Scenario 2: Separate Databases + Periodic Sync

Each person:
1. Has their own local database
2. Periodically share backups

```bash
# Person A exports
python backup_database.py

# Send file to Person B

# Person B imports
python restore_database.py database_backup_20260205.json
```

### Scenario 3: Git + Database Migrations

For schema changes (not data):
1. Use migration tools like Alembic
2. Share schema changes via Git
3. Each person runs migrations locally

## 🔒 Security for Cloud Deployment

### Critical: Never commit these

- ❌ `.env` (contains passwords)
- ❌ Database backup files with real user data
- ❌ `*.log` files

### Use Environment Variables

Cloud platforms provide environment variable settings:

**Heroku**:
```bash
heroku config:set DATABASE_URL=postgresql://...
```

**Render**: Set in dashboard under "Environment" tab

**Railway**: Set in dashboard under "Variables" tab

## 🔄 Continuous Deployment

### GitHub → Cloud (Automatic)

1. **Connect GitHub to your cloud platform**
   - Heroku: Enable GitHub integration
   - Render: Connect GitHub repo
   - Railway: Connect GitHub repo

2. **Every git push automatically deploys**
   ```bash
   git push origin main  # Automatically deploys to cloud
   ```

## 📈 Monitoring Production Database

### Check database health

```bash
# Heroku
heroku pg:info

# Or via Python script
python -c "from database import get_prediction_statistics; print(get_prediction_statistics())"
```

### Regular backups

```bash
# Run weekly backups
python backup_database.py

# Store backups securely (not in Git!)
```

## 🚨 What NOT to Do

1. ❌ **Don't** put database backups in Git (contains user data)
2. ❌ **Don't** share database passwords in public repos
3. ❌ **Don't** use local database for production (no backups, not accessible)
4. ❌ **Don't** give database access to untrusted users

## ✅ Best Practices

1. ✅ Use cloud database for production
2. ✅ Keep development and production databases separate
3. ✅ Regular backups (automated if possible)
4. ✅ Use environment variables for all credentials
5. ✅ Monitor database size and performance

## 📚 Recommended Reading

- [Heroku Postgres Guide](https://devcenter.heroku.com/categories/postgres)
- [12-Factor App Methodology](https://12factor.net/)
- [Database Security Best Practices](https://cheatsheetseries.owasp.org/cheatsheets/Database_Security_Cheat_Sheet.html)
