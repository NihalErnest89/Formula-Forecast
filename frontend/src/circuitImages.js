/**
 * Circuit image mappings for F1 tracks
 * Using actual circuit photos (aerial/track shots) similar to F1 TV
 * Images sourced from F1 official site Racehub Header Images
 */
export const circuitImages = {
  'Abu Dhabi Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Abu%20Dhabi.jpg.transform/9col/image.jpg',
  'Qatar Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Qatar.jpg.transform/9col/image.jpg',
  'Las Vegas Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Las%20Vegas.jpg.transform/9col/image.jpg',
  'São Paulo Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Brazil.jpg.transform/9col/image.jpg',
  'Sao Paulo Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Brazil.jpg.transform/9col/image.jpg',
  'Mexico City Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Mexico.jpg.transform/9col/image.jpg',
  'United States Grand Prix': 'https://media.formula1.com/image/upload/c_lfill,w_3392/q_auto/v1740000000/content/dam/fom-website/2018-redesign-assets/Racehub%20header%20images%2016x9/USA.webp',
  'USA Grand Prix': 'https://media.formula1.com/image/upload/c_lfill,w_3392/q_auto/v1740000000/content/dam/fom-website/2018-redesign-assets/Racehub%20header%20images%2016x9/USA.webp',
  'US Grand Prix': 'https://media.formula1.com/image/upload/c_lfill,w_3392/q_auto/v1740000000/content/dam/fom-website/2018-redesign-assets/Racehub%20header%20images%2016x9/USA.webp',
  'Singapore Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Singapore.jpg.transform/9col/image.jpg',
  'Azerbaijan Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Azerbaijan.jpg.transform/9col/image.jpg',
  'Italian Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Italy.jpg.transform/9col/image.jpg',
  'Dutch Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Netherlands.jpg.transform/9col/image.jpg',
  'Hungarian Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Hungary.jpg.transform/9col/image.jpg',
  'Belgian Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Belgium.jpg.transform/9col/image.jpg',
  'British Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Great%20Britain.jpg.transform/9col/image.jpg',
  'Austrian Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Austria.jpg.transform/9col/image.jpg',
  'Canadian Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Canada.jpg.transform/9col/image.jpg',
  'Spanish Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Spain.jpg.transform/9col/image.jpg',
  'Monaco Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Monaco.jpg.transform/9col/image.jpg',
  'Emilia Romagna Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Emilia%20Romagna.jpg.transform/9col/image.jpg',
  'Miami Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Miami.jpg.transform/9col/image.jpg',
  'Saudi Arabian Grand Prix': 'https://media.formula1.com/image/upload/c_lfill,w_3392/q_auto/v1740000000/content/dam/fom-website/2018-redesign-assets/Racehub%20header%20images%2016x9/Saudi_Arabia.webp',
  'Saudi Arabia Grand Prix': 'https://media.formula1.com/image/upload/c_lfill,w_3392/q_auto/v1740000000/content/dam/fom-website/2018-redesign-assets/Racehub%20header%20images%2016x9/Saudi_Arabia.webp',
  'Bahrain Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Bahrain.jpg.transform/9col/image.jpg',
  'Japanese Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Japan.jpg.transform/9col/image.jpg',
  'Chinese Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/China.jpg.transform/9col/image.jpg',
  'Australian Grand Prix': 'https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racehub%20Header%20Images%2016x9/Australia.jpg.transform/9col/image.jpg',
};

export const getCircuitImage = (eventName) => {
  if (!eventName) return null;
  
  // Direct match first
  if (circuitImages[eventName]) {
    return circuitImages[eventName];
  }
  
  // Case-insensitive match
  const lowerEventName = eventName.toLowerCase();
  for (const [key, value] of Object.entries(circuitImages)) {
    if (key.toLowerCase() === lowerEventName) {
      return value;
    }
  }
  
  // Partial match for variations (e.g., "Saudi Arabia" vs "Saudi Arabian")
  for (const [key, value] of Object.entries(circuitImages)) {
    const keyWords = key.toLowerCase().split(' ');
    const eventWords = lowerEventName.split(' ');
    // Check if most words match (at least 2 words or all words if less than 3)
    const matchingWords = eventWords.filter(word => keyWords.includes(word));
    if (matchingWords.length >= Math.min(2, keyWords.length)) {
      return value;
    }
  }
  
  return null;
};

