import React, { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, useMap } from 'react-leaflet';
import L, { DivIcon } from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { MapLocation } from '../types';
import { Navigation } from 'lucide-react';

// Custom colored leaflet markers — created lazily (inside render, not module scope)
const makeSvgIcon = (color: string): DivIcon => {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 36" width="24" height="36"><path fill="${color}" stroke="#0F172A" stroke-width="1.5" d="M12 0C5.37 0 0 5.37 0 12c0 9 12 24 12 24s12-15 12-24C24 5.37 18.63 0 12 0zm0 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"/></svg>`;
  return L.divIcon({
    className: '',
    html: svg,
    iconSize: [24, 36],
    iconAnchor: [12, 36],
    popupAnchor: [0, -36],
  });
};

const ICON_COLORS: Record<string, string> = {
  Critical: '#EF4444',
  High: '#F59E0B',
  Medium: '#EAB308',
  Low: '#10B981',
};

const getMarkerIcon = (priorityLevel: string): DivIcon =>
  makeSvgIcon(ICON_COLORS[priorityLevel] ?? '#10B981');

interface LeafletMapProps {
  locations?: MapLocation[];
  selectedLocation?: { latitude: number; longitude: number };
  onSelectLocation?: (lat: number, lng: number) => void;
  onSelectIncident?: (id: number) => void;
  center?: [number, number];
  zoom?: number;
  interactiveSelect?: boolean;
}

const LocationPickerHelper: React.FC<{ onSelect: (lat: number, lng: number) => void }> = ({
  onSelect,
}) => {
  useMapEvents({ click: (e) => onSelect(e.latlng.lat, e.latlng.lng) });
  return null;
};

const ChangeView: React.FC<{ center: [number, number]; zoom: number }> = ({ center, zoom }) => {
  const map = useMap();
  useEffect(() => { map.setView(center, zoom); }, [center, zoom, map]);
  return null;
};

export const LeafletMap: React.FC<LeafletMapProps> = ({
  locations = [],
  selectedLocation,
  onSelectLocation,
  onSelectIncident,
  center = [20.5937, 78.9629],
  zoom = 5,
  interactiveSelect = false,
}) => {
  return (
    <div className="relative w-full h-full min-h-[300px] rounded-xl overflow-hidden">
      <MapContainer
        center={center}
        zoom={zoom}
        scrollWheelZoom={false}
        style={{ width: '100%', height: '100%', minHeight: '300px' }}
      >
        <ChangeView center={center} zoom={zoom} />

        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {interactiveSelect && onSelectLocation && (
          <LocationPickerHelper onSelect={onSelectLocation} />
        )}

        {selectedLocation && (
          <Marker
            position={[selectedLocation.latitude, selectedLocation.longitude]}
            icon={getMarkerIcon('Critical')}
          >
            <Popup>
              <div className="p-1 text-xs">
                <p className="font-bold">Selected GPS Location</p>
                <p className="font-mono mt-1">
                  {selectedLocation.latitude.toFixed(4)}, {selectedLocation.longitude.toFixed(4)}
                </p>
              </div>
            </Popup>
          </Marker>
        )}

        {locations.map((loc) => (
          <Marker
            key={loc.id}
            position={[loc.latitude, loc.longitude]}
            icon={getMarkerIcon(loc.priority_level)}
            eventHandlers={{
              click: () => onSelectIncident && onSelectIncident(loc.id),
            }}
          >
            <Popup>
              <div className="p-2 space-y-1 text-xs min-w-[180px]">
                <div className="flex items-center justify-between border-b pb-1">
                  <span className="font-bold">{loc.title}</span>
                  <span className="font-mono font-bold uppercase text-[10px]">
                    {loc.priority_level}
                  </span>
                </div>
                <p>Type: <strong>{loc.disaster_type}</strong></p>
                <p>Score: <strong>{loc.priority_score.toFixed(1)}</strong></p>
                <p>Status: <strong>{loc.status}</strong></p>
                {onSelectIncident && (
                  <button
                    onClick={() => onSelectIncident(loc.id)}
                    className="w-full mt-2 py-1 bg-blue-600 text-white text-[11px] rounded font-semibold"
                  >
                    View Details
                  </button>
                )}
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
};
