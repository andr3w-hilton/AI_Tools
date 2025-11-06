# Droplet Control PWA

A Progressive Web App to control your DigitalOcean droplet from your phone. Start and stop your droplet with a single tap.

## Features

- Control any DigitalOcean droplet from your phone
- Works as a Progressive Web App (add to home screen)
- Secure - API token stored locally on your device
- Real-time droplet status monitoring
- Clean, modern interface

## Setup

1. Visit the app in your mobile browser
2. Tap "Share" → "Add to Home Screen"
3. Open the app and enter:
   - **API Token**: Generate one at https://cloud.digitalocean.com/account/api/tokens
   - **Droplet ID**: Find this in your droplet's URL (e.g., `cloud.digitalocean.com/droplets/123456789` → ID is `123456789`)
   - **App Password**: Create any password you'll remember

## How to Find Your Droplet ID

1. Log in to DigitalOcean
2. Go to your Droplets page
3. Click on your droplet
4. Look at the URL in your browser - the number at the end is your Droplet ID
   - Example: `https://cloud.digitalocean.com/droplets/123456789` → Droplet ID is `123456789`

## Security

- Your API token is stored locally in your browser only
- No data is sent to any third-party servers
- All communication is directly with DigitalOcean's API

## License

Free to use and modify for personal use.
