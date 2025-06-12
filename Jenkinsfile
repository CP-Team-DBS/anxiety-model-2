pipeline {
  agent any

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Build Docker Image') {
      steps {
        sh 'docker-compose build'
      }
    }

    stage('Create Docker Network') {
      steps {
        sh '''
          if ! docker network inspect anoto-network >/dev/null 2>&1; then
            docker network create anoto-network
          fi
        '''
      }
    }

    stage('Deploy') {
      steps {
        sh '''
          docker-compose down || true
          docker-compose up -d

          sleep 10
          docker ps | grep anoto-journal
        '''
      }
    }
  }

  post {
    failure {
      echo 'Pipeline failed'
      sh 'docker-compose down'
    }

    success {
      echo 'Deployment successful'
    }
  }
}
