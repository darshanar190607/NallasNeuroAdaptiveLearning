// System Integration Test Script
import fetch from 'node-fetch';

const API_BASE = 'http://localhost:5000/api';
const BCI_BASE = 'http://localhost:8000';

async function testSystemIntegration() {
  console.log('🧪 Testing NeuroAdaptive EdTech System Integration...\n');

  // Test 1: Backend Health Check
  try {
    console.log('1️⃣ Testing Backend Server...');
    const response = await fetch(`${API_BASE.replace('/api', '')}/`);
    const text = await response.text();
    console.log('✅ Backend:', text);
  } catch (error) {
    console.log('❌ Backend not running:', error.message);
  }

  // Test 2: BCI Service Health Check
  try {
    console.log('\n2️⃣ Testing BCI Service...');
    const response = await fetch(`${BCI_BASE}/health`);
    const data = await response.json();
    console.log('✅ BCI Service:', data);
  } catch (error) {
    console.log('❌ BCI Service not running:', error.message);
  }

  // Test 3: BCI Simulation
  try {
    console.log('\n3️⃣ Testing BCI Simulation...');
    const response = await fetch(`${BCI_BASE}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const data = await response.json();
    console.log('✅ BCI Simulation:', data);
  } catch (error) {
    console.log('❌ BCI Simulation failed:', error.message);
  }

  // Test 4: User Registration (Mock)
  try {
    console.log('\n4️⃣ Testing User Registration...');
    const testUser = {
      email: `test_${Date.now()}@example.com`,
      password: 'testpass123',
      name: 'Test User'
    };
    
    const response = await fetch(`${API_BASE}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(testUser)
    });
    
    if (response.ok) {
      const data = await response.json();
      console.log('✅ User Registration:', data.message);
      
      // Test 5: Survey Submission
      console.log('\n5️⃣ Testing Survey Submission...');
      const surveyData = {
        responses: {
          ageGroup: '18-22',
          educationLevel: 'undergraduate',
          learningStyle: 'visual',
          preferredPace: 'moderate',
          primaryInterests: ['Technology', 'Science'],
          careerGoals: ['Software Developer'],
          learningChallenges: ['None'],
          accommodationsNeeded: ['None'],
          deviceComfort: 'advanced',
          preferredContentTypes: ['videos', 'interactive-simulations'],
          studyEnvironment: 'quiet-room',
          preferredStudyTime: 'morning',
          sessionDuration: '45-60min',
          breakFrequency: 'every-30min',
          bciInterest: 'very-interested',
          adaptationComfort: 'moderate',
          privacyConcerns: 'minimal'
        }
      };
      
      const surveyResponse = await fetch(`${API_BASE}/survey/submit`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${data.token}`
        },
        body: JSON.stringify(surveyData)
      });
      
      if (surveyResponse.ok) {
        const surveyResult = await surveyResponse.json();
        console.log('✅ Survey Submission:', surveyResult.message);
        console.log('📊 Recommendations generated:', Object.keys(surveyResult.recommendations));
      } else {
        console.log('❌ Survey submission failed');
      }
      
    } else {
      const error = await response.json();
      console.log('❌ Registration failed:', error.message);
    }
  } catch (error) {
    console.log('❌ Registration test failed:', error.message);
  }

  console.log('\n🎉 System Integration Test Complete!');
  console.log('\n📋 Next Steps:');
  console.log('1. Start all services: run start_complete_system.bat');
  console.log('2. Open browser: http://localhost:5173');
  console.log('3. Register new account and complete survey');
  console.log('4. Experience adaptive learning with BCI integration');
}

// Run if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  testSystemIntegration().catch(console.error);
}

export { testSystemIntegration };